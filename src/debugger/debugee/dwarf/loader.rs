use crate::debugger::debugee::dwarf::EndianArcSlice;
use crate::debugger::error::Error;
use gimli::{
    AbbreviationsCache, DebugAbbrev, DebugAddr, DebugAranges, DebugInfo, DebugLine, DebugLineStr,
    DebugLoc, DebugLocLists, DebugMacinfo, DebugMacro, DebugNames, DebugRanges, DebugRngLists,
    DebugStr, DebugStrOffsets, DebugTypes, Dwarf, DwarfFileType, LocationLists, RangeLists,
    RunTimeEndian, Section, SectionId,
};
use object::{File, Object, ObjectSection};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::Mutex;

/// List of required sections for create a [`gimli::Dwarf`]
#[derive(Default)]
struct Sections {
    debug_abbrev: Option<DebugAbbrev<EndianArcSlice>>,
    debug_addr: Option<DebugAddr<EndianArcSlice>>,
    debug_aranges: Option<DebugAranges<EndianArcSlice>>,
    debug_info: Option<DebugInfo<EndianArcSlice>>,
    debug_line: Option<DebugLine<EndianArcSlice>>,
    debug_line_str: Option<DebugLineStr<EndianArcSlice>>,
    debug_macro: Option<DebugMacro<EndianArcSlice>>,
    debug_macinfo: Option<DebugMacinfo<EndianArcSlice>>,
    debug_names: Option<DebugNames<EndianArcSlice>>,
    debug_str: Option<DebugStr<EndianArcSlice>>,
    debug_str_offsets: Option<DebugStrOffsets<EndianArcSlice>>,
    debug_types: Option<DebugTypes<EndianArcSlice>>,
    debug_loc: Option<DebugLoc<EndianArcSlice>>,
    debug_loclists: Option<DebugLocLists<EndianArcSlice>>,
    debug_ranges: Option<DebugRanges<EndianArcSlice>>,
    debug_rnglists: Option<DebugRngLists<EndianArcSlice>>,
}

pub fn load_section(
    id: SectionId,
    file: &File,
    endian: RunTimeEndian,
) -> Result<EndianArcSlice, Error> {
    let data = file
        .section_by_name(id.name())
        .and_then(|section| section.uncompressed_data().ok())
        .unwrap_or(Cow::Borrowed(&[]));
    Ok(gimli::EndianArcSlice::new(Arc::from(&*data), endian))
}

/// Create a function that load section and put in [`Sections`] struct in right place.
macro_rules! make_sect_loader {
    ($file: expr, $endian: expr, $field: tt) => {{
        move |dest: Arc<Mutex<Option<Sections>>>| -> Result<(), Error> {
            let sect = Section::load(|id| load_section(id, $file, $endian))?;
            let mut lock = dest.lock().expect("unexpected: panic in another lock");
            let sections = lock.as_mut().expect("unexpected: sections must exists");
            sections.$field = Some(sect);
            Ok(())
        }
    }};
}

/// Load debug information from file. For better loading time all sections
/// loads in parallel inside a thread pool.
///
/// # Arguments
///
/// * `file`: object file with debug information
/// * `endian`: file endian
pub fn load_par(file: &File, endian: RunTimeEndian) -> Result<Dwarf<EndianArcSlice>, Error> {
    let load_debug_abbrev = make_sect_loader!(file, endian, debug_abbrev);
    let load_debug_addr = make_sect_loader!(file, endian, debug_addr);
    let load_debug_aranges = make_sect_loader!(file, endian, debug_aranges);
    let load_debug_info = make_sect_loader!(file, endian, debug_info);
    let load_debug_line = make_sect_loader!(file, endian, debug_line);
    let load_debug_line_str = make_sect_loader!(file, endian, debug_line_str);
    let load_debug_macro = make_sect_loader!(file, endian, debug_macro);
    let load_debug_macinfo = make_sect_loader!(file, endian, debug_macinfo);
    let load_debug_names = make_sect_loader!(file, endian, debug_names);
    let load_debug_str = make_sect_loader!(file, endian, debug_str);
    let load_debug_str_offsets = make_sect_loader!(file, endian, debug_str_offsets);
    let load_debug_types = make_sect_loader!(file, endian, debug_types);
    let load_debug_loc = make_sect_loader!(file, endian, debug_loc);
    let load_debug_loclists = make_sect_loader!(file, endian, debug_loclists);
    let load_debug_ranges = make_sect_loader!(file, endian, debug_ranges);
    let load_debug_rnglists = make_sect_loader!(file, endian, debug_rnglists);

    type SectLoaders<'a> =
        Vec<Box<dyn FnOnce(Arc<Mutex<Option<Sections>>>) -> Result<(), Error> + Send + Sync + 'a>>;

    let loaders: SectLoaders = vec![
        Box::new(load_debug_abbrev),
        Box::new(load_debug_addr),
        Box::new(load_debug_aranges),
        Box::new(load_debug_info),
        Box::new(load_debug_line),
        Box::new(load_debug_line_str),
        Box::new(load_debug_macro),
        Box::new(load_debug_macinfo),
        Box::new(load_debug_names),
        Box::new(load_debug_str),
        Box::new(load_debug_str_offsets),
        Box::new(load_debug_types),
        Box::new(load_debug_loc),
        Box::new(load_debug_loclists),
        Box::new(load_debug_ranges),
        Box::new(load_debug_rnglists),
    ];

    let sections = Arc::new(Mutex::new(Some(Sections::default())));
    loaders
        .into_par_iter()
        .try_for_each(|loader| loader(sections.clone()))?;

    // at this moment all sections must be loaded
    let sections = sections
        .lock()
        .expect("unexpected: panic in another lock")
        .take()
        .expect("unexpected: sections must exists");

    const SECT_MUST_EXISTS: &str = "section must exists";
    Ok(Dwarf {
        debug_abbrev: sections.debug_abbrev.expect(SECT_MUST_EXISTS),
        debug_addr: sections.debug_addr.expect(SECT_MUST_EXISTS),
        debug_aranges: sections.debug_aranges.expect(SECT_MUST_EXISTS),
        debug_info: sections.debug_info.expect(SECT_MUST_EXISTS),
        debug_line: sections.debug_line.expect(SECT_MUST_EXISTS),
        debug_line_str: sections.debug_line_str.expect(SECT_MUST_EXISTS),
        debug_macro: sections.debug_macro.expect(SECT_MUST_EXISTS),
        debug_macinfo: sections.debug_macinfo.expect(SECT_MUST_EXISTS),
        debug_names: sections.debug_names.expect(SECT_MUST_EXISTS),
        debug_str: sections.debug_str.expect(SECT_MUST_EXISTS),
        debug_str_offsets: sections.debug_str_offsets.expect(SECT_MUST_EXISTS),
        debug_types: sections.debug_types.expect(SECT_MUST_EXISTS),
        locations: LocationLists::new(
            sections.debug_loc.expect(SECT_MUST_EXISTS),
            sections.debug_loclists.expect(SECT_MUST_EXISTS),
        ),
        ranges: RangeLists::new(
            sections.debug_ranges.expect(SECT_MUST_EXISTS),
            sections.debug_rnglists.expect(SECT_MUST_EXISTS),
        ),
        file_type: DwarfFileType::Main,
        sup: None,
        abbreviations_cache: AbbreviationsCache::new(),
    })
}

pub fn load_dwo_section(
    id: SectionId,
    file: &File,
    endian: RunTimeEndian,
) -> Result<EndianArcSlice, Error> {
    let sect = id
        .dwo_name()
        .and_then(|name| file.section_by_name(name))
        .or_else(|| file.section_by_name(id.name()));

    let data = sect
        .and_then(|section| section.uncompressed_data().ok())
        .unwrap_or(Cow::Borrowed(&[]));
    Ok(gimli::EndianArcSlice::new(Arc::from(&*data), endian))
}

pub fn resolve_dwo_path(
    obj_path: &std::path::Path,
    comp_dir: Option<&str>,
    dwo_name: &str,
) -> Option<std::path::PathBuf> {
    let dwo_path = std::path::Path::new(dwo_name);

    if dwo_path.is_absolute() && dwo_path.exists() {
        return Some(dwo_path.to_path_buf());
    }

    if let Some(comp_dir) = comp_dir {
        let candidate = std::path::Path::new(comp_dir).join(dwo_path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Some(obj_dir) = obj_path.parent() {
        let candidate = obj_dir.join(dwo_path);
        if candidate.exists() {
            return Some(candidate);
        }

        if let Some(file_name) = dwo_path.file_name() {
            let candidate_deps = obj_dir.join("deps").join(file_name);
            if candidate_deps.exists() {
                return Some(candidate_deps);
            }

            let candidate_same = obj_dir.join(file_name);
            if candidate_same.exists() {
                return Some(candidate_same);
            }
        }
    }

    None
}

pub fn load_dwo(
    path: &std::path::Path,
    parent_dwarf: &Dwarf<EndianArcSlice>,
    expected_dwo_id: Option<gimli::DwoId>,
    skeleton_unit: &gimli::Unit<EndianArcSlice>,
) -> Result<(Dwarf<EndianArcSlice>, gimli::Unit<EndianArcSlice>), Error> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let obj = object::File::parse(&*mmap)?;
    let endian = if obj.is_little_endian() {
        RunTimeEndian::Little
    } else {
        RunTimeEndian::Big
    };

    let mut dwo_dwarf = Dwarf {
        debug_abbrev: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_addr: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_aranges: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_info: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_line: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_line_str: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_macro: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_macinfo: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_names: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_str: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_str_offsets: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        debug_types: Section::load(|id| load_dwo_section(id, &obj, endian))?,
        locations: LocationLists::new(
            Section::load(|id| load_dwo_section(id, &obj, endian))?,
            Section::load(|id| load_dwo_section(id, &obj, endian))?,
        ),
        ranges: RangeLists::new(
            Section::load(|id| load_dwo_section(id, &obj, endian))?,
            Section::load(|id| load_dwo_section(id, &obj, endian))?,
        ),
        file_type: DwarfFileType::Dwo,
        sup: None,
        abbreviations_cache: AbbreviationsCache::new(),
    };

    dwo_dwarf.make_dwo(parent_dwarf);

    let mut units_iter = dwo_dwarf.units();
    let mut chosen_unit = None;

    while let Some(header) = units_iter.next()? {
        if let Some(expected_id) = expected_dwo_id {
            match header.type_() {
                gimli::UnitType::SplitCompilation(id) if id == expected_id => {
                    let mut u = dwo_dwarf.unit(header)?;
                    u.copy_relocated_attributes(skeleton_unit);
                    chosen_unit = Some(u);
                    break;
                }
                _ => {
                    let u = dwo_dwarf.unit(header)?;
                    if u.dwo_id == Some(expected_id) {
                        let mut u = u;
                        u.copy_relocated_attributes(skeleton_unit);
                        chosen_unit = Some(u);
                        break;
                    }
                }
            }
        } else {
            let mut u = dwo_dwarf.unit(header)?;
            u.copy_relocated_attributes(skeleton_unit);
            chosen_unit = Some(u);
            break;
        }
    }

    let unit = chosen_unit.ok_or(Error::UnitNotFound(0_u64.into()))?;
    Ok((dwo_dwarf, unit))
}
