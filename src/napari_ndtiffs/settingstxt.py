"""Functions for parsing LLS Settings.txt files"""
import logging
import os
import re
import zipfile
from collections import defaultdict, namedtuple
from configparser import ConfigParser
from pathlib import Path

import dateutil.parser as dp

logger = logging.getLogger(__name__)


# repating pattern definitions used for parsing settings file
numstk_regx = re.compile(
    r"""
    \#\sof\sstacks\s\((?P<channel>\d)\) # channel number inside parentheses
    \s:\s+(?P<numstacks_requested>\d+)  # number of stacks after the colon
    """,
    re.MULTILINE | re.VERBOSE,
)

wavfrm_regx = re.compile(
    r"""
    ^(?P<waveform>.*)\sOffset,  # Waveform type, newline followed by description
    .*\((?P<channel>\d+)\)\s    # get channel number inside of parentheses
    :\s*(?P<offset>[-\d]*\.?\d*)    # float offset value after colon
    \s*(?P<interval>[-\d]*\.?\d*)   # float interval value next
    \s*(?P<numpix>\d+)          # integer number of pixels last
    """,
    re.MULTILINE | re.VERBOSE,
)

exc_regx = re.compile(
    r"""
    Excitation\sFilter,\s+Laser,    # Waveform type, newline followed by description
    .*\((?P<channel>\d+)\)\s    # get channel number inside of parentheses
    :\s+(?P<exfilter>[^\s]*)        # excitation filter: anything but whitespace
    \s+(?P<laser>\d+)           # integer laser line
    \s+(?P<power>\d*\.?\d*)     # float laser power value next
    \s+(?P<exposure>\d*\.?\d*)  # float exposure time last
    """,
    re.MULTILINE | re.VERBOSE,
)

PIXEL_SIZE = {"C11440-22C": 6.5, "C11440": 6.5, "C13440": 6.5}


class LLSSettingsParserError(Exception):
    pass


def parse_settings(path, pattern="*Settings.txt"):
    """Parse LLS Settings.txt file and return dict of info"""
    path = Path(path)
    if path.is_dir():
        sfiles = list(path.glob(pattern))
        if not sfiles:
            return {}
        if len(sfiles) > 1:
            logger.warn("Multiple Settings.txt files detected. " "Using first one.")
        path = sfiles[0]
    if not path.is_file():
        raise FileNotFoundError(f"Could not read file: {str(path)}")
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as z:
                settext = next(
                    i
                    for i in z.namelist()
                    if i.endswith("Settings.txt")
                    and not os.path.basename(i).startswith(".")
                )
                with z.open(settext) as f:
                    text = f.read().decode()
        except StopIteration as e:
            raise FileNotFoundError(
                f"Could not find Settings.txt in archive {str(path)}"
            ) from e
    else:
        with open(str(path), encoding="utf-8") as f:
            text = f.read()

    sections = [t.strip() for t in re.split(r"(?:[\*\s]+)([^\*]+)", text) if t.strip()]
    if len(sections) % 2:
        raise LLSSettingsParserError("Section headings not properly parsed")
    sections = dict(zip(sections[::2], sections[1::2]))
    for k in ("General", "Waveform", "Camera"):
        if k not in sections:
            raise LLSSettingsParserError(
                "Cannot parse settings file without " '"{}"" section'.format(k)
            )

    def _search(regex, default=None, func=lambda x: x, section=text):
        match = re.search(regex, sections[section])
        return func(match[1].strip()) if match and len(match.groups()) else default

    _D = dict(params={}, camera={}, channels=defaultdict(lambda: defaultdict(dict)))

    # basic stuff from General section
    searches = [
        # Section     label    regex               default         formatter
        ("General", ("date", r"Date\s*:\s*(.*)\n", None, dp.parse)),
        ("General", ("acq_mode", r"Acq Mode\s*:\s*(.*)\n")),
        ("General", ("software_version", r"Version\s*:\s*v ([\d*.?]+)")),
        ("Waveform", ("cycle_lasers", r"Cycle lasers\s*:\s*(.*)(?:$|\n)")),
        ("Waveform", ("z_motion", r"Z motion\s*:\s*(.*)(?:$|\n)")),
    ]
    for section, item in searches:
        key, *patrn = item
        _D[key] = _search(*patrn, section=section)

    # channel-specific information
    for regx in (wavfrm_regx, exc_regx, numstk_regx):
        for item in regx.finditer(sections["Waveform"]):
            i = item.groupdict()
            c = int(i.pop("channel"))
            w = i.pop("waveform", False)
            if w:
                _D["channels"][c][w].update(numberdict(i))
            else:
                _D["channels"][c].update(numberdict(i))

    # camera section
    cp = ConfigParser(strict=False)
    cp.read_string("[Section]\n" + sections["Camera"])
    cp = cp[cp.sections()[0]]
    for s in ("model", "serial", "exp(s)", "cycle(s)", "cycle(hz)", "roi"):
        _D["camera"].update({re.sub(r"(\()(.+)(\))", r"_\2", s): cp.get(s)})
    if _D["camera"].get("roi"):
        # [left, top, right, bottom]
        _D["camera"]["roi"] = [int(q) for q in re.findall(r"\d+", cp.get("roi"))]

    try:
        _D["camera"]["pixel"] = PIXEL_SIZE[_D["camera"]["model"].split("-")[0]]
    except Exception:
        # relatively safe assumption
        _D["camera"]["pixel"] = 6.5

    # general .ini File section

    # parse the ini part
    cp = ConfigParser(strict=False)
    cp.optionxform = str  # leave case in keys
    cp.read_string(sections[".ini File"])
    # not everyone will have added Annular mask to their settings ini
    inner, outer = (None, None)
    for n in ["Mask", "Annular Mask", "Annulus"]:
        if cp.has_section(n):
            for k, v in cp["Annular Mask"].items():
                if "inner" in k:
                    inner = float(v)
                if "outer" in k:
                    outer = float(v)
    mask = None
    if inner is not None and outer is not None:
        mask = namedtuple("Mask", ["inner", "outer"])(inner, outer)
    _D["params"]["mask"] = mask

    _D["params"]["angle"] = cp.getfloat(
        "Sample stage", "Angle between stage and bessel beam (deg)"
    )
    _D["mag"] = cp.getfloat("Detection optics", "Magnification")
    _D["camera"]["name"] = cp.get("General", "Camera type")
    _D["camera"]["trigger_mode"] = cp.get("General", "Cam Trigger mode")
    _D["mag"] = cp.getfloat("Detection optics", "Magnification")
    _D["camera"]["twincam"] = cp.getboolean("General", "Twin cam mode?")

    try:
        _D["camera"]["cam2_name"] = cp.get("General", "2nd Camera type")
    except Exception:
        _D["camera"]["cam2_name"] = "Disabled"
    _D["ini"] = cp

    _D["params"]["nc"] = len(_D["channels"])
    # only update this from the data folder
    # if _D['channels'][0]['numstacks_requested']:
    #     _D['params']['nt'] = int(_D['channels'][0]['numstacks_requested'])
    # else:
    #     _D['params']['nt'] = None
    _D["params"]["nx"] = None  # .camera.roi.height
    _D["params"]["ny"] = None  # .camera.roi.width
    _D["params"]["nz"] = None
    if _D["camera"]["roi"]:
        left, top, right, bottom = _D["camera"]["roi"]
        _D["params"]["nx"] = abs(right - left) + 1
        _D["params"]["ny"] = abs(bottom - top) + 1

    _D["params"]["dx"] = None
    _D["params"]["dz"] = None
    if _D["camera"]["pixel"] and _D["mag"]:
        _D["params"]["dx"] = round(_D["camera"]["pixel"] / _D["mag"], 4)

    _D["params"]["wavelengths"] = [
        _D["channels"][v]["laser"] for v in sorted(_D["channels"].keys())
    ]
    _D["params"]["samplescan"] = _D["z_motion"] == "Sample piezo"
    _D["params"]["roi"] = _D["camera"]["roi"]

    k = "S PZT" if _D["z_motion"] == "Sample piezo" else "Z PZT"
    try:
        _D["params"]["nz"] = _D["channels"][0][k]["numpix"]
        _D["params"]["dz"] = abs(_D["channels"][0][k]["interval"])
    except Exception:
        _D["params"]["dz"] = None

    _D["channels"] = {k: dict(v) for k, v in _D["channels"].items()}
    return _D


def numberdict(dct):
    """convert all numeric values in a dict to their appropriate type"""
    o = {}
    for k, v in dct.items():
        if v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                v = v
        o[k] = v
    return o
