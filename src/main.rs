use anyhow::{anyhow, Context, Result};
use clap::Parser;
use fuser::{
    Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation, INodeNo,
    LockOwner, MountOption, OpenFlags, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, Request,
    SessionACL,
};
use lru::LruCache;
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{self, File},
    io::{BufReader, Read, Seek},
    num::NonZeroUsize,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use chd::metadata::{KnownMetadata, Metadata, MetadataTag};
use chd::Chd;

/// Expose 2048-byte ISO stream from CD CHDs and passthrough from DVD CHDs.
const TTL: Duration = Duration::from_secs(1);
const CD_FRAME_2352: usize = 2352;
const CD_FRAME_2448: usize = 2448; // 2352-byte raw sector + 96-byte subchannel (chdman createcd output)

/// VCD (POPSTARTER/OPL) format: a 1 MiB header followed by the raw 2352-byte disc image.
/// Header layout matches the cue2pops v2.0 specification (see <https://github.com/leji-a/psx-vcd>).
const VCD_HEADER_SIZE: usize = 0x10_0000; // 1 MiB
const VCD_PREGAP_SECTORS: u32 = 150; // 2 seconds at 75 sectors/second
/// cue2pops v2.0 signature ("kHn ") written at offset 0x400.
const VCD_SIGNATURE: [u8; 4] = [0x6B, 0x48, 0x6E, 0x20];

/// Flags / CLI
#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_PKG_NAME"),
    author,
    version,
    about = env!("CARGO_PKG_DESCRIPTION"),
    long_about = None
)]
struct Args {
    /// Source directory containing *.chd files
    #[arg(short = 's', long = "source", value_name = "DIR")]
    source_dir: PathBuf,

    /// Mountpoint
    #[arg(short = 'm', long = "mount", value_name = "DIR")]
    mountpoint: PathBuf,

    /// Allow other users to access the mount (requires user_allow_other in /etc/fuse.conf)
    #[arg(long = "allow-other", default_value_t = false)]
    allow_other: bool,

    /// Max in-memory cache entries (frames) across all files
    #[arg(long = "cache-hunks", default_value_t = 256)]
    cache_hunks: usize,

    /// Soft cap for cache memory usage (bytes)
    #[arg(long = "cache-bytes", default_value_t = 256 * 1024 * 1024)]
    cache_bytes: usize,

    /// Max number of simultaneously open CHD file handles
    #[arg(long = "max-open-chds", default_value_t = 8)]
    max_open_chds: usize,

    /// Permit exporting Mode2/Form2 payloads as raw 2324-byte sectors (exposed as "Name (Form2).bin")
    #[arg(long = "cd-allow-form2", default_value_t = false)]
    cd_allow_form2: bool,

    /// Serve CD (PlayStation) CHDs as POPSTARTER/OPL-compatible .vcd files instead of .iso
    #[arg(long = "vcd", default_value_t = false)]
    vcd: bool,

    /// Recurse into subdirectories when scanning for *.chd files
    #[arg(long = "recursive", default_value_t = false)]
    recursive: bool,

    /// Verbose logging
    #[arg(long = "verbose", default_value_t = false)]
    verbose: bool,
}

#[derive(Clone, Debug)]
enum BackingKind {
    /// DVD (or generic 2048 units): direct 2048 sector passthrough
    Dvd2048,
    /// CD-style frames (2352 or 2448 bytes) -> 2048-byte user-data view
    Cd2352 {
        first_data_lba: u64,
        payload_kind: CdPayloadKind,
        track_frames: Option<u64>,
        /// Raw frame size in bytes: 2352 (no subchannel) or 2448 (with subchannel, chdman createcd)
        frame_bytes: usize,
    },
    /// Raw/unrecognized, default to 2048 passthrough (rare/fallback)
    Raw2048,
    /// PlayStation CD served as a POPSTARTER/OPL VCD: a 1 MiB header followed by
    /// the raw 2352-byte disc image (all tracks, in order).
    Vcd {
        /// Number of 2352-byte sectors in the disc image (body length / 2352).
        total_frames: u64,
        /// Raw frame size stored in the CHD: 2352 (no subchannel) or 2448 (with subchannel).
        frame_bytes: usize,
        /// Track table used to build the VCD TOC header.
        tracks: Arc<Vec<TrackInfo>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CdPayloadKind {
    Mode1_2048,
    Mode2Form1_2048,
    Mode2Form2_2324,
}

#[derive(Clone, Debug)]
struct IndexEntry {
    ino: u64,
    name: String,
    chd_path: PathBuf,
    kind: BackingKind,
    iso_size: u64,
}

struct Handle {
    file_id: u64,
    chd_path: PathBuf,
}

type ChdHandle = Arc<Mutex<Chd<BufReader<File>>>>;

struct FsState {
    args: Args,
    entries: Vec<IndexEntry>,
    handles: Mutex<HashMap<u64, Handle>>,
    next_fh: Mutex<u64>,
    /// Open CHD handles, keyed by file_id (inode). Avoids reopening on every read().
    /// Capped at 64 entries; LRU eviction closes the file descriptor.
    open_chds: Mutex<LruCache<u64, ChdHandle>>,
    frame_cache: Mutex<LruCache<(u64, u64), Vec<u8>>>,
    approx_cache_bytes: Mutex<usize>,
    /// Cached 1 MiB VCD headers, keyed by file_id (inode). Built lazily on first read.
    vcd_header_cache: Mutex<LruCache<u64, Arc<Vec<u8>>>>,
}

impl FsState {
    fn new(args: Args) -> Result<Self> {
        let cache_cap =
            NonZeroUsize::new(args.cache_hunks).unwrap_or(NonZeroUsize::new(64).unwrap());

        let open_chds_cap =
            NonZeroUsize::new(args.max_open_chds).unwrap_or(NonZeroUsize::new(8).unwrap());

        Ok(Self {
            entries: Vec::new(),
            handles: Mutex::new(HashMap::new()),
            next_fh: Mutex::new(1),
            open_chds: Mutex::new(LruCache::new(open_chds_cap)),
            frame_cache: Mutex::new(LruCache::new(cache_cap)),
            approx_cache_bytes: Mutex::new(0),
            vcd_header_cache: Mutex::new(LruCache::new(open_chds_cap)),
            args,
        })
    }

    /// Return a cached open CHD handle for the given file_id, opening it if needed.
    fn get_chd(&self, file_id: u64, path: &Path) -> Result<ChdHandle> {
        let mut map = self.open_chds.lock().expect("open_chds poisoned");
        if let Some(h) = map.get(&file_id) {
            return Ok(Arc::clone(h));
        }
        let f = File::open(path)?;
        let chd = Chd::open(BufReader::new(f), None)?;
        let handle = Arc::new(Mutex::new(chd));
        // Arc::clone before insert so the evicted entry (if any) is dropped after we release
        // the map lock, not while holding it.
        let ret = Arc::clone(&handle);
        map.put(file_id, handle);
        Ok(ret)
    }

    fn build_index(&mut self) -> Result<()> {
        let dir = self.args.source_dir.clone();
        let mut tmp: Vec<IndexEntry> = Vec::new();

        let mut dirs: Vec<PathBuf> = vec![dir.clone()];
        while let Some(current) = dirs.pop() {
            for ent in fs::read_dir(&current).with_context(|| format!("reading {current:?}"))? {
                let ent = ent?;
                let path = ent.path();
                let ft = ent.file_type()?;

                if ft.is_dir() {
                    if self.args.recursive {
                        dirs.push(path);
                    }
                    continue;
                }

                if path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("chd"))
                    != Some(true)
                {
                    continue;
                }

                match self.build_index_entry(&path) {
                    Ok(Some((name, kind, size))) => {
                        tmp.push(IndexEntry {
                            ino: 0,
                            name,
                            chd_path: path.clone(),
                            kind,
                            iso_size: size,
                        });
                    }
                    Ok(None) => {}
                    Err(e) => {
                        error!("Skipping {:?}: {}", path, e);
                    }
                }
            }
        }

        tmp.sort_by_key(|a| a.name.to_lowercase());

        for (i, e) in tmp.iter_mut().enumerate() {
            e.ino = (i as u64) + 2;
        }

        self.entries = tmp;
        Ok(())
    }

    fn build_index_entry(&self, chd_path: &Path) -> Result<Option<(String, BackingKind, u64)>> {
        let f = File::open(chd_path)?;
        let mut chd = Chd::open(BufReader::new(f), None)?;

        let hdr = chd.header();
        let unit_bytes = hdr.unit_bytes() as usize;
        let logical_bytes = hdr.logical_bytes();

        let stem = chd_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        if unit_bytes == 2048 {
            let iso_size = logical_bytes;
            let name = format!("{stem}.iso");
            return Ok(Some((name, BackingKind::Dvd2048, iso_size)));
        }

        if unit_bytes == 2352 {
            let total_frames = logical_bytes / 2352;

            if self.args.vcd {
                return self.build_vcd_entry(
                    &mut chd,
                    chd_path,
                    stem,
                    CD_FRAME_2352,
                    total_frames,
                );
            }

            if let Some((first_lba, payload, track_frames)) = {
                let mut rf = BufReader::new(File::open(chd_path)?);
                parse_cd_toc_from_metadata(&mut chd, &mut rf, self.args.cd_allow_form2)?
            } {
                let (per_sector, name) = match payload {
                    CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => {
                        (2048u64, format!("{stem}.iso"))
                    }
                    CdPayloadKind::Mode2Form2_2324 => {
                        if self.args.cd_allow_form2 {
                            (2324u64, format!("{stem} (Form2).bin"))
                        } else {
                            return Ok(None);
                        }
                    }
                };

                let frames = track_frames.unwrap_or(total_frames - first_lba);
                let iso_size = frames * per_sector;
                let kind = BackingKind::Cd2352 {
                    first_data_lba: first_lba,
                    payload_kind: payload,
                    track_frames,
                    frame_bytes: CD_FRAME_2352,
                };

                return Ok(Some((name, kind, iso_size)));
            }

            let (first_lba, payload) = quick_scan_first_data(
                &mut chd,
                total_frames,
                self.args.cd_allow_form2,
                CD_FRAME_2352,
            )?;

            let (per_sector, name) = match payload {
                CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => {
                    (2048u64, format!("{stem}.iso"))
                }
                CdPayloadKind::Mode2Form2_2324 => {
                    if self.args.cd_allow_form2 {
                        (2324u64, format!("{stem} (Form2).bin"))
                    } else {
                        return Ok(None);
                    }
                }
            };

            let iso_size = (total_frames - first_lba) * per_sector;
            let kind = BackingKind::Cd2352 {
                first_data_lba: first_lba,
                payload_kind: payload,
                track_frames: None,
                frame_bytes: CD_FRAME_2352,
            };

            return Ok(Some((name, kind, iso_size)));
        }

        if unit_bytes == CD_FRAME_2448 {
            let total_frames = logical_bytes / CD_FRAME_2448 as u64;

            if self.args.vcd {
                return self.build_vcd_entry(
                    &mut chd,
                    chd_path,
                    stem,
                    CD_FRAME_2448,
                    total_frames,
                );
            }

            if let Some((first_lba, payload, track_frames)) = {
                let mut rf = BufReader::new(File::open(chd_path)?);
                parse_cd_toc_from_metadata(&mut chd, &mut rf, self.args.cd_allow_form2)?
            } {
                let (per_sector, name) = match payload {
                    CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => {
                        (2048u64, format!("{stem}.iso"))
                    }
                    CdPayloadKind::Mode2Form2_2324 => {
                        if self.args.cd_allow_form2 {
                            (2324u64, format!("{stem} (Form2).bin"))
                        } else {
                            return Ok(None);
                        }
                    }
                };

                let frames = track_frames.unwrap_or(total_frames - first_lba);
                let iso_size = frames * per_sector;
                let kind = BackingKind::Cd2352 {
                    first_data_lba: first_lba,
                    payload_kind: payload,
                    track_frames,
                    frame_bytes: CD_FRAME_2448,
                };

                return Ok(Some((name, kind, iso_size)));
            }

            let (first_lba, payload) = quick_scan_first_data(
                &mut chd,
                total_frames,
                self.args.cd_allow_form2,
                CD_FRAME_2448,
            )?;

            let (per_sector, name) = match payload {
                CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => {
                    (2048u64, format!("{stem}.iso"))
                }
                CdPayloadKind::Mode2Form2_2324 => {
                    if self.args.cd_allow_form2 {
                        (2324u64, format!("{stem} (Form2).bin"))
                    } else {
                        return Ok(None);
                    }
                }
            };

            let iso_size = (total_frames - first_lba) * per_sector;
            let kind = BackingKind::Cd2352 {
                first_data_lba: first_lba,
                payload_kind: payload,
                track_frames: None,
                frame_bytes: CD_FRAME_2448,
            };

            return Ok(Some((name, kind, iso_size)));
        }

        let name = format!("{stem}.iso");
        Ok(Some((name, BackingKind::Raw2048, logical_bytes)))
    }

    fn alloc_fh(&self) -> u64 {
        let mut next_fh = self.next_fh.lock().expect("next_fh mutex poisoned");
        let fh = *next_fh;
        *next_fh += 1;
        fh
    }

    #[allow(clippy::too_many_arguments)]
    fn read_iso_from_cd(
        &self,
        file_id: u64,
        path: &Path,
        start_frame: u64,
        payload_kind: CdPayloadKind,
        offset: u64,
        size: u32,
        max_len: u64,
        frame_bytes: usize,
        reply: ReplyData,
    ) {
        let per_sector = match payload_kind {
            CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => 2048usize,
            CdPayloadKind::Mode2Form2_2324 => 2324usize,
        };

        let payload_start = match payload_kind {
            CdPayloadKind::Mode1_2048 => 16usize,
            CdPayloadKind::Mode2Form1_2048 => 24usize,
            CdPayloadKind::Mode2Form2_2324 => 24usize,
        };

        if offset >= max_len || size == 0 {
            reply.data(&[]);
            return;
        }

        let end = offset.saturating_add(size as u64).min(max_len);

        let mut want = end - offset;
        let mut out = Vec::with_capacity(want as usize);
        let mut cur_iso_sector = offset / per_sector as u64;
        let mut cur_in_sector_off = offset % per_sector as u64;

        while want > 0 {
            let frame_idx = start_frame + cur_iso_sector;
            let sec = match self.get_cd_frame(file_id, path, frame_idx, frame_bytes) {
                Ok(v) => v,
                Err(e) => {
                    error!("frame read error: {:?}", e);
                    reply.error(Errno::from_i32(libc::EIO));
                    return;
                }
            };

            let payload = &sec[payload_start..payload_start + per_sector];
            let avail = per_sector as u64 - cur_in_sector_off;
            let take = avail.min(want);

            out.extend_from_slice(
                &payload[cur_in_sector_off as usize..(cur_in_sector_off + take) as usize],
            );

            want -= take;
            cur_iso_sector += 1;
            cur_in_sector_off = 0;
        }

        reply.data(&out);
    }

    fn get_cd_frame(
        &self,
        file_id: u64,
        path: &Path,
        frame_index: u64,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        {
            let mut cache = self.frame_cache.lock().expect("frame_cache mutex poisoned");
            if let Some(buf) = cache.get(&(file_id, frame_index)) {
                return Ok(buf.clone());
            }
        }

        let chd_handle = self.get_chd(file_id, path)?;
        let mut chd = chd_handle.lock().expect("chd handle poisoned");

        let hunk_bytes = chd.header().hunk_size() as usize;
        let frames_per_hunk = hunk_bytes / frame_bytes;

        if frames_per_hunk == 0 {
            return Err(anyhow!("invalid hunk size for CD"));
        }

        let hunk_index = (frame_index as usize) / frames_per_hunk;
        let frame_in_hunk = (frame_index as usize) % frames_per_hunk;

        let mut hunk_buf = chd.get_hunksized_buffer();
        let mut cmp_buf = Vec::new();

        let mut hk = chd.hunk(hunk_index as u32)?;
        hk.read_hunk_in(&mut cmp_buf, &mut hunk_buf)?;

        let frame_off = frame_in_hunk * frame_bytes;
        // Always return the 2352-byte raw sector; subchannel bytes (2352..2448, if any) are dropped.
        let owned = hunk_buf[frame_off..frame_off + CD_FRAME_2352].to_vec();

        drop(chd); // release lock before acquiring frame_cache lock

        {
            let mut cache = self.frame_cache.lock().expect("frame_cache mutex poisoned");
            let mut approx_cache_bytes = self
                .approx_cache_bytes
                .lock()
                .expect("approx_cache_bytes mutex poisoned");

            *approx_cache_bytes += owned.len();

            while *approx_cache_bytes > self.args.cache_bytes {
                if let Some((_k, v)) = cache.pop_lru() {
                    *approx_cache_bytes = approx_cache_bytes.saturating_sub(v.len());
                } else {
                    break;
                }
            }

            cache.put((file_id, frame_index), owned.clone());
        }

        Ok(owned)
    }

    /// Build a VCD index entry for a CD CHD: a 1 MiB POPSTARTER/OPL header followed
    /// by the raw 2352-byte disc image (all tracks, in order).
    fn build_vcd_entry(
        &self,
        chd: &mut Chd<BufReader<File>>,
        chd_path: &Path,
        stem: &str,
        frame_bytes: usize,
        total_frames: u64,
    ) -> Result<Option<(String, BackingKind, u64)>> {
        let mut rf = BufReader::new(File::open(chd_path)?);
        let mut tracks = parse_all_tracks_from_metadata(chd, &mut rf)?;

        // No CD track metadata: assume a single Mode2 data track spanning the disc.
        if tracks.is_empty() {
            tracks.push(TrackInfo {
                number: 1,
                kind: TrackKind::Mode2Form1,
                frames: total_frames as u32,
                pregap: 0,
                postgap: 0,
            });
        }

        let name = format!("{stem}.vcd");
        let size = VCD_HEADER_SIZE as u64 + total_frames * CD_FRAME_2352 as u64;
        let kind = BackingKind::Vcd {
            total_frames,
            frame_bytes,
            tracks: Arc::new(tracks),
        };

        Ok(Some((name, kind, size)))
    }

    /// Return the cached 1 MiB VCD header for the given file, building it on first use.
    fn get_vcd_header(
        &self,
        file_id: u64,
        tracks: &[TrackInfo],
        total_frames: u64,
    ) -> Arc<Vec<u8>> {
        let mut cache = self
            .vcd_header_cache
            .lock()
            .expect("vcd_header_cache mutex poisoned");

        if let Some(h) = cache.get(&file_id) {
            return Arc::clone(h);
        }

        let header = Arc::new(build_vcd_header(tracks, total_frames));
        let ret = Arc::clone(&header);
        cache.put(file_id, header);
        ret
    }
}

/// Parse CD TOC from CHD metadata (CHTR/CHT2). Returns (first_data_lba, payload_kind, frames_in_track).
fn parse_cd_toc_from_metadata<R: Read + Seek>(
    chd: &mut Chd<R>,
    file: &mut R,
    allow_form2: bool,
) -> Result<Option<(u64, CdPayloadKind, Option<u64>)>> {
    let mut tracks: Vec<TrackInfo> = Vec::new();

    let it = chd.metadata_refs();
    for mref in it {
        let md: Metadata = mref.read(file)?;
        let tag = md.metatag;

        if tag != KnownMetadata::CdRomTrack.metatag() && tag != KnownMetadata::CdRomTrack2.metatag()
        {
            continue;
        }

        let s = String::from_utf8_lossy(&md.value).to_string();
        if let Some(ti) = parse_track_line(&s) {
            tracks.push(ti);
        }
    }

    if tracks.is_empty() {
        return Ok(None);
    }

    tracks.sort_by_key(|t| t.number);

    let mut lba: u64 = 0;
    for t in &tracks {
        lba += t.pregap as u64;

        let payload = match t.kind {
            TrackKind::Audio => None,
            TrackKind::Mode1 => Some(CdPayloadKind::Mode1_2048),
            TrackKind::Mode2Form1 => Some(CdPayloadKind::Mode2Form1_2048),
            TrackKind::Mode2Form2 => {
                if allow_form2 {
                    Some(CdPayloadKind::Mode2Form2_2324)
                } else {
                    None
                }
            }
            TrackKind::Mode2Raw => Some(CdPayloadKind::Mode2Form1_2048), // raw sectors: default to Form1 (PS2 data track)
        };

        if let Some(pk) = payload {
            let frames_in_track = t.frames as u64;
            return Ok(Some((lba, pk, Some(frames_in_track))));
        }

        lba += t.frames as u64;
        lba += t.postgap as u64;
    }

    Ok(None)
}

#[derive(Debug, Clone)]
struct TrackInfo {
    number: u32,
    kind: TrackKind,
    frames: u32,
    pregap: u32,
    postgap: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackKind {
    Audio,
    Mode1,
    Mode2Form1,
    Mode2Form2,
    Mode2Raw,
}

/// Parse every CD track described in the CHD metadata (CHTR/CHT2), sorted by track number.
fn parse_all_tracks_from_metadata<R: Read + Seek>(
    chd: &mut Chd<R>,
    file: &mut R,
) -> Result<Vec<TrackInfo>> {
    let mut tracks: Vec<TrackInfo> = Vec::new();

    for mref in chd.metadata_refs() {
        let md: Metadata = mref.read(file)?;
        let tag = md.metatag;

        if tag != KnownMetadata::CdRomTrack.metatag() && tag != KnownMetadata::CdRomTrack2.metatag()
        {
            continue;
        }

        let s = String::from_utf8_lossy(&md.value).to_string();
        if let Some(ti) = parse_track_line(&s) {
            tracks.push(ti);
        }
    }

    tracks.sort_by_key(|t| t.number);
    Ok(tracks)
}

/// Encode a value (0..=99) as a single Binary-Coded-Decimal byte.
fn to_bcd(v: u8) -> u8 {
    ((v / 10) << 4) | (v % 10)
}

/// Convert an LBA sector count into a 3-byte BCD MSF (Minutes:Seconds:Frames).
fn msf_bcd_from_sectors(sectors: u32) -> [u8; 3] {
    let frames = sectors % 75;
    let total_seconds = sectors / 75;
    let seconds = total_seconds % 60;
    let minutes = total_seconds / 60;
    [to_bcd(minutes as u8), to_bcd(seconds as u8), to_bcd(frames as u8)]
}

/// Build the 1 MiB VCD header (cue2pops v2.0 layout) for a disc with the given track
/// table and total raw-sector count. See <https://github.com/leji-a/psx-vcd>.
fn build_vcd_header(tracks: &[TrackInfo], total_frames: u64) -> Vec<u8> {
    let mut h = vec![0u8; VCD_HEADER_SIZE];

    let total_sectors = total_frames as u32;
    let track_count = tracks.len().clamp(1, 99) as u8;

    let first_is_audio = tracks.first().map(|t| t.kind == TrackKind::Audio).unwrap_or(false);
    let last_is_audio = tracks.last().map(|t| t.kind == TrackKind::Audio).unwrap_or(false);
    let content_type = if last_is_audio { 0x01 } else { 0x41 };

    // Descriptor A0 (first track / disc type).
    h[0] = if first_is_audio { 0x01 } else { 0x41 };
    h[2] = 0xA0;
    h[7] = 0x01; // first track number
    h[8] = 0x20; // CD-XA disc type

    // Descriptor A1 (last track / content type).
    h[10] = content_type;
    h[12] = 0xA1;
    h[17] = to_bcd(track_count);
    h[20] = content_type;

    // Descriptor A2 (lead-out): cue2pops adds 150 sectors for the lead-out MSF.
    h[22] = 0xA2;
    let leadout = msf_bcd_from_sectors(total_sectors + VCD_PREGAP_SECTORS);
    h[27] = leadout[0];
    h[28] = leadout[1];
    h[29] = leadout[2];

    // Track entries (10 bytes each), starting at offset 0x1E, using cue2pops v2.0 MSF math.
    let mut accumulated: u32 = 0;
    let mut offset = 30;
    for (i, t) in tracks.iter().enumerate() {
        if offset + 10 > 1024 {
            break; // never overrun into the signature/sector-count area
        }

        h[offset] = if t.kind == TrackKind::Audio { 0x01 } else { 0x41 };
        h[offset + 2] = to_bcd((t.number % 100) as u8);

        let (index00_sector, index01_sector) = if i == 0 {
            (0u32, VCD_PREGAP_SECTORS)
        } else if t.pregap > 0 {
            let i00 = accumulated + VCD_PREGAP_SECTORS + VCD_PREGAP_SECTORS;
            (i00, i00 + VCD_PREGAP_SECTORS)
        } else {
            let s = accumulated + VCD_PREGAP_SECTORS;
            (s, s)
        };

        h[offset + 3..offset + 6].copy_from_slice(&msf_bcd_from_sectors(index00_sector));
        h[offset + 7..offset + 10].copy_from_slice(&msf_bcd_from_sectors(index01_sector));

        accumulated += t.frames;
        offset += 10;
    }

    // cue2pops v2.0 signature at 0x400 and total sector counts at 0x408 / 0x40C.
    h[1024..1028].copy_from_slice(&VCD_SIGNATURE);
    let sector_bytes = total_sectors.to_le_bytes();
    h[1032..1036].copy_from_slice(&sector_bytes);
    h[1036..1040].copy_from_slice(&sector_bytes);

    h
}

#[cfg(feature = "doccheck")]
fn dump_all_flags_and_exit() -> ! {
    use clap::CommandFactory;
    use std::process;

    let cmd = <Args as CommandFactory>::command();
    let mut flags: Vec<String> = Vec::new();

    for arg in cmd.get_arguments() {
        if let Some(long) = arg.get_long() {
            flags.push(format!("--{}", long));
        }
    }

    flags.sort();
    flags.dedup();

    for f in flags {
        println!("{f}");
    }

    process::exit(0);
}

fn parse_track_line(s: &str) -> Option<TrackInfo> {
    let mut number = None;
    let mut frames = 0u32;
    let mut pregap = 0u32;
    let mut postgap = 0u32;
    let mut kind = None::<TrackKind>;

    for tok in s.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }

        if let Some((k, v)) = tok.split_once(':') {
            match k {
                "TRACK" => number = v.parse().ok(),
                "FRAMES" => frames = v.parse().unwrap_or(0),
                "PREGAP" => pregap = v.parse().unwrap_or(0),
                "POSTGAP" => postgap = v.parse().unwrap_or(0),
                "TYPE" => {
                    kind = Some(match v {
                        "MODE1" => TrackKind::Mode1,
                        "MODE2/2048" | "MODE2_FORM1" => TrackKind::Mode2Form1,
                        "MODE2/2324" | "MODE2_FORM2" => TrackKind::Mode2Form2,
                        "MODE2/2352" | "MODE2_RAW" | "CDI/2352" => TrackKind::Mode2Raw,
                        "AUDIO" => TrackKind::Audio,
                        other => {
                            if other.starts_with("MODE2") && other.contains("2048") {
                                TrackKind::Mode2Form1
                            } else if other.starts_with("MODE2") && other.contains("2324") {
                                TrackKind::Mode2Form2
                            } else {
                                TrackKind::Audio
                            }
                        }
                    })
                }
                _ => {}
            }
        }
    }

    Some(TrackInfo {
        number: number?,
        kind: kind?,
        frames,
        pregap,
        postgap,
    })
}

/// Fallback when metadata is missing: scan early frames to find a data sector.
fn quick_scan_first_data<R: Read + Seek>(
    chd: &mut Chd<R>,
    total_frames: u64,
    allow_form2: bool,
    frame_bytes: usize,
) -> Result<(u64, CdPayloadKind)> {
    let scan_limit = total_frames.min(2000);
    let mut cmp = Vec::new();
    let mut hbuf = chd.get_hunksized_buffer();
    let frames_per_hunk = (chd.header().hunk_size() as usize) / frame_bytes;

    let mut frame: u64 = 0;
    while frame < scan_limit {
        let hunk_index = (frame as usize) / frames_per_hunk;
        let frame_in_hunk = (frame as usize) % frames_per_hunk;

        let mut hk = chd.hunk(hunk_index as u32)?;
        hk.read_hunk_in(&mut cmp, &mut hbuf)?;

        let base = frame_in_hunk * frame_bytes;
        // Inspect only the 2352-byte raw sector; subchannel bytes (if any) are beyond this range.
        let sec = &hbuf[base..base + CD_FRAME_2352];

        let mode = sec[0x0F];

        if mode == 0x01 {
            return Ok((frame, CdPayloadKind::Mode1_2048));
        } else if mode == 0x02 {
            // Determine Form 1 vs Form 2 from the Submode byte (SM) in the sector subheader.
            // Subheader: bytes 16-23 of the raw sector; SM is byte 18.  Bit 5 (0x20) = Form 2.
            let sm = sec[18];
            if (sm & 0x20) != 0 && allow_form2 {
                return Ok((frame, CdPayloadKind::Mode2Form2_2324));
            } else {
                return Ok((frame, CdPayloadKind::Mode2Form1_2048));
            }
        }

        frame += 1;
    }

    Ok((0, CdPayloadKind::Mode1_2048))
}

impl Filesystem for FsState {
    fn lookup(&self, _req: &Request, _parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let name_str = name.to_string_lossy().to_string();

        if let Some(e) = self.entries.iter().find(|e| e.name == name_str) {
            let attr = file_attr_for(e).unwrap_or_else(|_| default_file_attr(e));
            reply.entry(&TTL, &attr, Generation(0));
        } else {
            reply.error(Errno::from_i32(libc::ENOENT));
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, fh: Option<FileHandle>, reply: ReplyAttr) {
        let _ = fh;

        if ino.0 == 1 {
            let attr = FileAttr {
                ino: INodeNo(1),
                size: 0,
                blocks: 1,
                atime: SystemTime::now(),
                mtime: SystemTime::now(),
                ctime: SystemTime::now(),
                crtime: SystemTime::UNIX_EPOCH,
                kind: FileType::Directory,
                perm: 0o755,
                nlink: 2,
                uid: unsafe { libc::geteuid() },
                gid: unsafe { libc::getegid() },
                rdev: 0,
                flags: 0,
                blksize: 4096,
            };

            reply.attr(&TTL, &attr);
            return;
        }

        if let Some(e) = self.entries.iter().find(|e| e.ino == ino.0) {
            match file_attr_for(e) {
                Ok(attr) => reply.attr(&TTL, &attr),
                Err(_) => reply.error(Errno::from_i32(libc::EIO)),
            }
        } else {
            reply.error(Errno::from_i32(libc::ENOENT));
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        if ino.0 != 1 {
            reply.error(Errno::from_i32(libc::ENOTDIR));
            return;
        }

        let mut idx = offset;

        if idx == 0 {
            let _ = reply.add(INodeNo(1), 1, FileType::Directory, ".");
            let _ = reply.add(INodeNo(1), 2, FileType::Directory, "..");
            idx = 2;
        }

        let mut ent_idx = 3u64;
        for e in &self.entries {
            if ent_idx <= idx {
                ent_idx += 1;
                continue;
            }

            if reply.add(
                INodeNo(e.ino),
                ent_idx,
                FileType::RegularFile,
                e.name.as_str(),
            ) {
                break;
            }

            ent_idx += 1;
        }

        reply.ok();
    }

    fn open(&self, _req: &Request, ino: INodeNo, _flags: OpenFlags, reply: fuser::ReplyOpen) {
        let (file_id, chd_path) = if let Some(e) = self.entries.iter().find(|e| e.ino == ino.0) {
            (e.ino, e.chd_path.clone())
        } else {
            reply.error(Errno::from_i32(libc::ENOENT));
            return;
        };

        if File::open(&chd_path).is_err() {
            reply.error(Errno::from_i32(libc::EIO));
            return;
        }

        let fh = self.alloc_fh();

        self.handles
            .lock()
            .expect("handles mutex poisoned")
            .insert(fh, Handle { file_id, chd_path });

        reply.opened(FileHandle(fh), FopenFlags::empty());
    }

    fn release(
        &self,
        _req: &Request,
        _ino: INodeNo,
        fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: fuser::ReplyEmpty,
    ) {
        self.handles
            .lock()
            .expect("handles mutex poisoned")
            .remove(&fh.0);

        reply.ok();
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let ent = match self.entries.iter().find(|e| e.ino == ino.0) {
            Some(e) => e.clone(),
            None => {
                reply.error(Errno::from_i32(libc::ENOENT));
                return;
            }
        };

        if size == 0 {
            reply.data(&[]);
            return;
        }

        let (file_id, chd_path) = match self
            .handles
            .lock()
            .expect("handles mutex poisoned")
            .get(&fh.0)
        {
            Some(h) => (h.file_id, h.chd_path.clone()),
            None => {
                reply.error(Errno::from_i32(libc::EBADF));
                return;
            }
        };

        match ent.kind {
            BackingKind::Dvd2048 | BackingKind::Raw2048 => {
                let start = offset;

                if start >= ent.iso_size {
                    reply.data(&[]);
                    return;
                }

                let end = start.saturating_add(size as u64).min(ent.iso_size);
                let to_read = (end - start) as usize;

                let chd_handle = match self.get_chd(file_id, &chd_path) {
                    Ok(h) => h,
                    Err(_) => {
                        reply.error(Errno::from_i32(libc::EIO));
                        return;
                    }
                };
                let mut chd = chd_handle.lock().expect("chd handle poisoned");

                let hunk_size = chd.header().hunk_size() as u64;
                let mut buf = vec![0u8; to_read];
                let mut out_off = 0usize;
                let mut left = to_read as u64;
                let mut pos = start;

                while left > 0 {
                    let hunk_idx = (pos / hunk_size) as u32;
                    let in_hunk_off = (pos % hunk_size) as usize;
                    let take = ((hunk_size as usize) - in_hunk_off).min(left as usize);

                    let mut hunk_buf = chd.get_hunksized_buffer();
                    let mut cmp = Vec::new();

                    let mut hk = match chd.hunk(hunk_idx) {
                        Ok(h) => h,
                        Err(_) => {
                            reply.error(Errno::from_i32(libc::EIO));
                            return;
                        }
                    };

                    if hk.read_hunk_in(&mut cmp, &mut hunk_buf).is_err() {
                        reply.error(Errno::from_i32(libc::EIO));
                        return;
                    }

                    buf[out_off..out_off + take]
                        .copy_from_slice(&hunk_buf[in_hunk_off..in_hunk_off + take]);

                    out_off += take;
                    left -= take as u64;
                    pos += take as u64;
                }

                reply.data(&buf);
            }
            BackingKind::Cd2352 {
                first_data_lba,
                payload_kind,
                track_frames,
                frame_bytes,
            } => {
                let per_sector = match payload_kind {
                    CdPayloadKind::Mode1_2048 | CdPayloadKind::Mode2Form1_2048 => 2048u64,
                    CdPayloadKind::Mode2Form2_2324 => 2324u64,
                };

                let max_len = if let Some(fr) = track_frames {
                    fr * per_sector
                } else {
                    ent.iso_size
                };

                self.read_iso_from_cd(
                    file_id,
                    &chd_path,
                    first_data_lba,
                    payload_kind,
                    offset,
                    size,
                    max_len,
                    frame_bytes,
                    reply,
                );
            }
            BackingKind::Vcd {
                total_frames,
                frame_bytes,
                ref tracks,
            } => {
                let header_size = VCD_HEADER_SIZE as u64;
                let body_size = total_frames * CD_FRAME_2352 as u64;
                let total = header_size + body_size;

                if offset >= total {
                    reply.data(&[]);
                    return;
                }

                let end = offset.saturating_add(size as u64).min(total);
                let mut out = Vec::with_capacity((end - offset) as usize);
                let mut pos = offset;

                // Serve from the 1 MiB header region.
                if pos < header_size {
                    let header = self.get_vcd_header(file_id, tracks, total_frames);
                    let h_end = end.min(header_size);
                    out.extend_from_slice(&header[pos as usize..h_end as usize]);
                    pos = h_end;
                }

                // Serve the raw 2352-byte disc image that follows the header.
                while pos < end {
                    let body_off = pos - header_size;
                    let frame_idx = body_off / CD_FRAME_2352 as u64;
                    let in_frame = (body_off % CD_FRAME_2352 as u64) as usize;

                    let sec = match self.get_cd_frame(file_id, &chd_path, frame_idx, frame_bytes) {
                        Ok(v) => v,
                        Err(e) => {
                            error!("vcd frame read error: {:?}", e);
                            reply.error(Errno::from_i32(libc::EIO));
                            return;
                        }
                    };

                    let avail = CD_FRAME_2352 - in_frame;
                    let take = avail.min((end - pos) as usize);
                    out.extend_from_slice(&sec[in_frame..in_frame + take]);
                    pos += take as u64;
                }

                reply.data(&out);
            }
        }
    }
}

fn default_file_attr(e: &IndexEntry) -> FileAttr {
    FileAttr {
        ino: INodeNo(e.ino),
        size: e.iso_size,
        blocks: e.iso_size.div_ceil(512),
        atime: SystemTime::now(),
        mtime: SystemTime::now(),
        ctime: SystemTime::now(),
        crtime: SystemTime::UNIX_EPOCH,
        kind: FileType::RegularFile,
        perm: 0o444,
        nlink: 1,
        uid: unsafe { libc::geteuid() },
        gid: unsafe { libc::getegid() },
        rdev: 0,
        flags: 0,
        blksize: 4096,
    }
}

fn file_attr_for(e: &IndexEntry) -> Result<FileAttr> {
    let meta = e.chd_path.metadata()?;

    Ok(FileAttr {
        ino: INodeNo(e.ino),
        size: e.iso_size,
        blocks: e.iso_size.div_ceil(512),
        atime: SystemTime::now(),
        mtime: SystemTime::UNIX_EPOCH + Duration::from_secs(meta.mtime() as u64),
        ctime: SystemTime::UNIX_EPOCH + Duration::from_secs(meta.ctime() as u64),
        crtime: SystemTime::UNIX_EPOCH,
        kind: FileType::RegularFile,
        perm: 0o444,
        nlink: 1,
        uid: meta.uid(),
        gid: meta.gid(),
        rdev: 0,
        flags: 0,
        blksize: 4096,
    })
}

fn main() -> Result<()> {
    #[cfg(feature = "doccheck")]
    if std::env::args().any(|a| a == "--dump-flags") {
        dump_all_flags_and_exit();
    }

    let args = Args::parse();

    let filter = if args.verbose {
        EnvFilter::new("info")
    } else {
        EnvFilter::new("warn")
    };

    tracing_subscriber::fmt().with_env_filter(filter).init();

    if args.mountpoint.metadata().is_err() {
        return Err(anyhow!(
            "Mountpoint {:?} does not exist or is not accessible",
            args.mountpoint
        ));
    }

    let mut fs = FsState::new(args)?;
    fs.build_index()?;

    let mut config = Config::default();
    config.mount_options = vec![
        MountOption::FSName("chd2iso".into()),
        MountOption::RO,
        MountOption::DefaultPermissions,
    ];

    if fs.args.allow_other {
        config.acl = SessionACL::All;
        config.mount_options.push(MountOption::AutoUnmount);
    }

    info!(
        "mounting {:?} -> {:?} (entries: {})",
        fs.args.source_dir,
        fs.args.mountpoint,
        fs.entries.len()
    );

    let mountpoint = fs.args.mountpoint.clone();
    fuser::mount2(fs, &mountpoint, &config).map_err(|e| anyhow!("mount failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode1_track_line() {
        let line = "TRACK:1 TYPE:MODE1 SUBTYPE:NONE FRAMES:26888 PREGAP:0 PGTYPE:MODE1 PGSUB:RW_RAW POSTGAP:0";
        let ti = parse_track_line(line).expect("should parse MODE1 track");

        assert_eq!(ti.number, 1);
        assert_eq!(ti.kind, TrackKind::Mode1);
        assert_eq!(ti.frames, 26888);
        assert_eq!(ti.pregap, 0);
        assert_eq!(ti.postgap, 0);
    }

    #[test]
    fn parse_mode2_2048_track_line() {
        let line = "TRACK:2 TYPE:MODE2/2048 FRAMES:1234 PREGAP:5 POSTGAP:6";
        let ti = parse_track_line(line).expect("should parse MODE2/2048 track");

        assert_eq!(ti.number, 2);
        assert_eq!(ti.kind, TrackKind::Mode2Form1);
        assert_eq!(ti.frames, 1234);
        assert_eq!(ti.pregap, 5);
        assert_eq!(ti.postgap, 6);
    }

    #[test]
    fn parse_mode2_2324_track_line() {
        let line = "TRACK:3 TYPE:MODE2/2324 FRAMES:567 PREGAP:0 POSTGAP:0";
        let ti = parse_track_line(line).expect("should parse MODE2/2324 track");

        assert_eq!(ti.number, 3);
        assert_eq!(ti.kind, TrackKind::Mode2Form2);
        assert_eq!(ti.frames, 567);
    }

    #[test]
    fn parse_malformed_track_line() {
        let line = "TRACK:4 FRAMES:100";
        assert!(parse_track_line(line).is_none());
    }

    #[test]
    fn vcd_msf_bcd_conversion() {
        // 0 sectors -> 00:00:00
        assert_eq!(msf_bcd_from_sectors(0), [0x00, 0x00, 0x00]);
        // 150 sectors = 2 seconds -> 00:02:00
        assert_eq!(msf_bcd_from_sectors(150), [0x00, 0x02, 0x00]);
        // 1 minute 30 seconds 50 frames = (90*75 + 50) = 6800 sectors
        assert_eq!(msf_bcd_from_sectors(90 * 75 + 50), [0x01, 0x30, 0x50]);
    }

    #[test]
    fn vcd_header_single_data_track() {
        let tracks = vec![TrackInfo {
            number: 1,
            kind: TrackKind::Mode2Form1,
            frames: 10_000,
            pregap: 0,
            postgap: 0,
        }];
        let total_frames = 10_000u64;
        let h = build_vcd_header(&tracks, total_frames);

        assert_eq!(h.len(), VCD_HEADER_SIZE);

        // Descriptor A0 (data disc, CD-XA).
        assert_eq!(h[0], 0x41);
        assert_eq!(h[2], 0xA0);
        assert_eq!(h[7], 0x01);
        assert_eq!(h[8], 0x20);

        // Descriptor A1 (data content, one track).
        assert_eq!(h[10], 0x41);
        assert_eq!(h[12], 0xA1);
        assert_eq!(h[17], to_bcd(1));
        assert_eq!(h[20], 0x41);

        // Descriptor A2 lead-out = total + 150 sectors.
        assert_eq!(h[22], 0xA2);
        assert_eq!(
            [h[27], h[28], h[29]],
            msf_bcd_from_sectors(total_frames as u32 + 150)
        );

        // First track entry: type DATA, number 1, INDEX 00 = 0, INDEX 01 = 150.
        assert_eq!(h[30], 0x41);
        assert_eq!(h[32], to_bcd(1));
        assert_eq!([h[33], h[34], h[35]], msf_bcd_from_sectors(0));
        assert_eq!([h[37], h[38], h[39]], msf_bcd_from_sectors(150));

        // cue2pops signature and duplicated little-endian sector counts.
        assert_eq!(&h[1024..1028], &VCD_SIGNATURE);
        assert_eq!(&h[1032..1036], &(total_frames as u32).to_le_bytes());
        assert_eq!(&h[1036..1040], &(total_frames as u32).to_le_bytes());
    }

    #[test]
    fn vcd_header_data_plus_audio_track() {
        let tracks = vec![
            TrackInfo {
                number: 1,
                kind: TrackKind::Mode2Form1,
                frames: 20_000,
                pregap: 0,
                postgap: 0,
            },
            TrackInfo {
                number: 2,
                kind: TrackKind::Audio,
                frames: 5_000,
                pregap: 0,
                postgap: 0,
            },
        ];
        let total_frames = 25_000u64;
        let h = build_vcd_header(&tracks, total_frames);

        // Last track is audio -> A1 content type is CDDA (0x01).
        assert_eq!(h[10], 0x01);
        assert_eq!(h[17], to_bcd(2));
        assert_eq!(h[20], 0x01);

        // Track 2 entry (offset 40): audio type, number 2,
        // INDEX 00 = INDEX 01 = track1_frames + 150.
        assert_eq!(h[40], 0x01);
        assert_eq!(h[42], to_bcd(2));
        let expected = msf_bcd_from_sectors(20_000 + 150);
        assert_eq!([h[43], h[44], h[45]], expected);
        assert_eq!([h[47], h[48], h[49]], expected);
    }
}