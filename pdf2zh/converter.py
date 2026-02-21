import concurrent.futures
import logging
import re
import unicodedata
from enum import Enum
from string import Template
from typing import Dict

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    X302AITranslator,
)

log = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # Override replace cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # Override return command stream
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # Override set pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # Override return command stream
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # Override set cid and font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack insert original character encoding
        item.font = font  # hack insert original character font
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y  # initial y coordinate
        self.x: float = x  # initial x coordinate
        self.x0: float = x0  # left boundary
        self.x1: float = x1  # right boundary
        self.y0: float = y0  # top boundary
        self.y1: float = y1  # bottom boundary
        self.size: float = size  # font size
        self.brk: bool = brk  # line break marker


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
        force_font: bool = False,
        force_font_size: int = 100,
        bypass_parser: bool = False,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.force_font = force_font
        self.force_font_size = force_font_size
        self.bypass_parser = bypass_parser
        self.translator: BaseTranslator = None
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator, QwenMtTranslator, X302AITranslator]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # Paragraph
        sstk: list[str] = []            # paragraph text stack
        pstk: list[Paragraph] = []      # paragraph attribute stack
        vbkt: int = 0                   # paragraph formula bracket count
        # Formula group
        vstk: list[LTChar] = []         # formula symbol group
        vlstk: list[LTLine] = []        # formula line group
        vfix: float = 0                 # formula vertical offset
        # Formula group stack
        var: list[list[LTChar]] = []    # formula symbol group stack
        varl: list[list[LTLine]] = []   # formula line group stack
        varf: list[float] = []          # formula vertical offset stack
        vlen: list[float] = []          # formula width stack
        # Global
        lstk: list[LTLine] = []         # global line stack
        xt: LTChar = None               # previous character
        xt_cls: int = -1                # paragraph of previous character, ensures new paragraph can be triggered regardless of first character's category
        vmax: float = ltpage.width / 4  # max width of inline formula
        ops: str = ""                   # Render result

        def vflag(font: str, char: str):    # Match formula (and subscript/superscript) font
            if isinstance(font, bytes):     # May not be able to decode, convert to str directly
                try:
                    font = font.decode('utf-8')  # Try to decode using UTF-8
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # Font name truncation
            if re.match(r"\(cid:", char):
                return True
            # Judgment based on font name rules
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex font
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # Judgment based on character set rules
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # non-space
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # Text modifiers, math symbols, separators
                        or ord(char[0]) in range(0x370, 0x400)          # Greek letters
                    )
                ):
                    return True
            return False

        ############################################################
        # A. Original document parsing
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height might be height in fig, use layout.shape uniformly here
                h, w = layout.shape
                # Read category of current character in layout
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # Anchor bullet position in document
                if child.get_text() == "•":
                    cls = 0
                # Determine if current character belongs to formula
                is_formula_char = (                                                                         # Determine if current character belongs to formula
                    cls == 0                                                                                # 1. Category is reserved area
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. Subscript/superscript font, 0.76 for subscript and 0.799 for uppercase, use 0.79 as midpoint, also consider drop caps
                    or vflag(child.fontname, child.get_text())                                              # 3. Formula font
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. Vertical font
                )
                if is_formula_char:
                    cur_v = True
                if self.bypass_parser and is_formula_char:
                    cur_v = False
                # Determine if bracket group belongs to formula
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # Determine if current formula ends
                    not cur_v                                               # 1. Current character does not belong to formula
                    or cls != xt_cls                                        # 2. Current character does not belong to same paragraph as previous one
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. Line break within paragraph, could be a long italic paragraph or fraction break, set threshold to distinguish
                    # Prohibit line break in pure formula (code) paragraph, until text starts and restarts text paragraph, ensure only two cases exist
                    # A. Pure formula (code) paragraph (anchor absolute position) sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. Text starting paragraph (typesetting relative position) sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # Because cls==xt_cls==0 implies sstk[-1]=="", so no need to check cls!=0 here
                ):
                    if vstk:
                        if (                                                # Correct formula vertical offset based on text to the right of formula
                            not cur_v                                       # 1. Current character does not belong to formula
                            and cls == xt_cls                               # 2. Current character belongs to same paragraph as previous one
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. Current character is to the right of formula
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # Prohibit subsequent connection of pure formula paragraph (sstk[-1]=="{v*}"), but consider connection of new character and subsequent characters, so modify category of previous character here
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # Current character does not belong to formula or is the first character of formula
                if not vstk:
                    if child.get_text() in ["+", "-"]:
                        sstk[-1] += "\n"
                        pstk[-1].brk = True
                    elif cls == xt_cls:               # Current character belongs to same paragraph as previous one
                        if child.x0 > xt.x1 + 1:    # Add inline space
                            if child.get_text() in [" "]:
                                pass
                            else:
                                sstk[-1] += " "
                        elif child.x1 < xt.x0:      # Add line break space and mark original paragraph has line break
                            if self.bypass_parser and is_formula_char:
                                sstk.append("")
                                pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                            else:
                                sstk[-1] += "\n"
                                pstk[-1].brk = True
                    else:                           # Build a new paragraph based on current character
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                if not cur_v:                                               # Push text to stack
                    if (                                                    # Correct paragraph attributes based on current character
                        child.size > pstk[-1].size                          # 1. Current character is larger than paragraph font
                        or len(sstk[-1].strip()) == 1                       # 2. Current character is the second character of paragraph (consider drop caps)
                    ) and child.get_text() != " ":                          # 3. Current character is not space
                        pstk[-1].y -= child.size - pstk[-1].size            # Correct paragraph initial y coordinate, assuming top boundaries of two different sized characters align
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # Push formula to stack
                    if (                                                    # Correct formula vertical offset based on text to the left of formula
                        not vstk                                            # 1. Current character is the first character of formula
                        and cls == xt_cls                                   # 2. Current character belongs to same paragraph as previous one
                        and child.x0 > xt.x0                                # 3. Previous character is to the left of formula
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # Update paragraph boundaries, because line break within paragraph might be followed by formula start, so handle outside
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # Update previous character
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # Chart/Figure
                pass
            elif isinstance(child, LTLine):     # Line
                layout = self.layout[ltpage.pageid]
                # ltpage.height might be height in fig, use layout.shape uniformly here
                h, w = layout.shape
                # Read category of current line in layout
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # Formula line
                    vlstk.append(child)
                else:                           # Global line
                    lstk.append(child)
            else:
                pass
        # Handle end
        if vstk:    # Pop formula
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # Calculate formula width
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. Paragraph translation
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # Multi-threaded translation
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # Do not translate whitespace and formulas
                return s
            try:
                # Replace line breaks with spaces to prevent word concatenation
                # Line breaks in the source are visual PDF line wraps, not semantic breaks
                s = s.replace("\n", " ")
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, sstk))

        ############################################################
        # C. New document typesetting
        def raw_string(fcur: str, cstk: str):  # Encode string
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # Determine encoding length
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # Get default line height based on target language
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # Default 1.1 for minor languages
        # Languages that use space-based word wrapping (not character-based like CJK)
        SPACE_WRAP_LANGS = {"vi", "en", "fr", "de", "es", "pt", "it", "ru", "uk", "pl", "nl", "sv", "da", "no", "fi", "cs", "sk", "hu", "ro", "bg", "hr", "sl", "lt", "lv", "et", "tr", "id", "ms", "tl", "sw"}
        space_wrap = self.translator.lang_out.lower() in SPACE_WRAP_LANGS
        _x, _y = 0, 0
        ops_list = []

        def calc_char_width(ch, font_name):
            """Calculate the advance width of a single character given its font."""
            if font_name == self.noto_name:
                return self.noto.char_lengths(ch, size)[0]
            else:
                return self.fontmap[font_name].char_width(ord(ch)) * size

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # Paragraph initial x coordinate
            y: float = pstk[id].y                       # Paragraph initial y coordinate
            x0: float = pstk[id].x0                     # Paragraph left boundary
            x1: float = pstk[id].x1                     # Paragraph right boundary
            height: float = pstk[id].y1 - pstk[id].y0   # Paragraph height
            size: float = pstk[id].size * (self.force_font_size / 100.0) # Paragraph font size
            brk: bool = pstk[id].brk                    # Paragraph line break marker
            cstk: str = ""                              # Current text stack
            fcur: str = None                            # Current font ID
            lidx = 0                                    # Record line break count
            tx = x
            fcur_ = fcur
            ptr = 0
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []

            while ptr < len(new):
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # Match {vn} formula marker
                mod = 0  # Text modifier
                if vy_regex:  # Load formula
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # Translator might automatically add an out-of-bounds formula marker
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # Text modifier
                        mod = var[vid][-1].width
                else:  # Load text
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if (
                            not self.force_font
                            and fcur_ is None
                            and self.fontmap["tiro"].to_unichr(ord(ch)) == ch
                        ):
                            fcur_ = "tiro"  # Default Latin font
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # Default non-Latin font
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if (                                # Output text buffer
                    fcur_ != fcur                   # 1. Font update
                    or vy_regex                     # 2. Insert formula
                    or x + adv > x1 + 0.1 * size    # 3. Reach right boundary (possible that entire line is symbolized, need to consider floating point error involved)
                ):
                    if cstk:
                        # Word wrapping: if overflow and space_wrap, split at last space
                        if space_wrap and not vy_regex and fcur_ == fcur and x + adv > x1 + 0.1 * size:
                            space_idx = cstk.rfind(" ")
                            if space_idx > 0:
                                # Flush prefix (up to and including the space)
                                prefix = cstk[:space_idx]
                                suffix = cstk[space_idx + 1:]  # After the space
                                ops_vals.append({
                                    "type": OpType.TEXT,
                                    "font": fcur,
                                    "size": size,
                                    "x": tx,
                                    "dy": 0,
                                    "rtxt": raw_string(fcur, prefix),
                                    "lidx": lidx
                                })
                                # Move to next line
                                lidx += 1
                                x = x0
                                tx = x0
                                # Recalculate x for suffix
                                for sc in suffix:
                                    x += calc_char_width(sc, fcur)
                                cstk = suffix
                            else:
                                # No space found, flush entire buffer as-is
                                ops_vals.append({
                                    "type": OpType.TEXT,
                                    "font": fcur,
                                    "size": size,
                                    "x": tx,
                                    "dy": 0,
                                    "rtxt": raw_string(fcur, cstk),
                                    "lidx": lidx
                                })
                                cstk = ""
                        else:
                            ops_vals.append({
                                "type": OpType.TEXT,
                                "font": fcur,
                                "size": size,
                                "x": tx,
                                "dy": 0,
                                "rtxt": raw_string(fcur, cstk),
                                "lidx": lidx
                            })
                            cstk = ""
                if x + adv > x1 + 0.1 * size:  # Reach right boundary, wrap to next line
                    x = x0
                    lidx += 1
                if vy_regex:  # Insert formula
                    fix = 0
                    if fcur is not None:  # Correct vertical offset for formula within paragraph
                        fix = varf[vid]
                    for vch in var[vid]:  # Typeset formula character
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # Typeset formula line
                        if l.linewidth < 5:  # hack some documents use thick lines as image background
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # Insert into text buffer
                    if not cstk:  # Start of single line
                        tx = x
                        if x == x0 and ch == " ":  # Eliminate paragraph line break space
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # Text modifier
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # Handle end
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # Typeset global line
            if l.linewidth < 5:  # hack some documents use thick lines as image background
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
