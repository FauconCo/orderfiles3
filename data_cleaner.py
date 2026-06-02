%%writefile data_cleaner_v4.py
import polars as pl
import re
import unicodedata

class AssyrianDataCleaner:
    def __init__(self):
        self.fraction_map = {
            '0.5': '½', '0.25': '¼', '0.3333': '⅓', '0.8333': '⅚', 
            '0.625': '⅝', '0.6666': '⅔', '0.75': '¾', '0.1666': '⅙'
        }
        self.subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        self.roman_to_int = {
            'XII': '12', 'XI': '11', 'IX': '9', 'X': '10', 'VIII': '8', 'VII': '7', 
            'VI': '6', 'IV': '4', 'V': '5', 'III': '3', 'II': '2', 'I': '1'
        }
        self.re_float = re.compile(r'(\d+\.\d{4})\d+')

    def _pre_process_python(self, text, is_translation=False):
        if text is None:
            return "<manual_split_needed>"
        text = str(text)
        if not text.strip():
            return "<manual_split_needed>"
        
        try:
            text = unicodedata.normalize('NFC', text)
            
            if not is_translation:
                text = text.translate(self.subscript_map)
                
            text = self.re_float.sub(r'\1', text)
            
            if is_translation:
                for roman, arab in self.roman_to_int.items():
                    text = re.sub(f"(?i)\\b(month)\\s+{roman}\\b", f"\\g<1> {arab}", text)
                
            return text
        except Exception:
            return "<manual_split_needed>"

    def process_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        if 'transliteration' in df.columns:
            expr = pl.col('transliteration').map_elements(
                lambda x: self._pre_process_python(x, False), return_dtype=pl.String
            )
            
            expr = expr.str.replace_all(r'<\((.*?)\)>', r'${1}')
            expr = expr.str.replace_all(r'!(sv|pr|sg|sn|eq|wp|cs)\b', '')
            expr = expr.str.replace_all(r'%(a|h|eg|e)\b', '')
            
            expr = expr.str.replace_all(r'(^|\s)[#"~|=^@](\s|$)', ' ')
            expr = expr.str.replace_all(r'(^|\s)[#"~|=^@](\s|$)', ' ')
            
            gap_patterns = r'\[x\]|\bx\b|…|\.\.\.|\(break\)|\(large break\)|\(n broken lines\)'
            expr = expr.str.replace_all(gap_patterns, '<gap>')
            expr = expr.str.replace_all('<big_gap>', '<gap>', literal=True)
            expr = expr.str.replace_all(r'(?:<gap>[\s\-]*)+<gap>', '<gap>')
            expr = expr.str.replace_all(r'(<gap>\s*)+', '<gap> ')
            
            expr = expr.str.replace_all(r'[!?/:]', '')
            expr = expr.str.replace_all(r'<<', '')
            expr = expr.str.replace_all(r'>>', '')
            expr = expr.str.replace_all(r'\[', '')
            expr = expr.str.replace_all(r'\]', '')
            expr = expr.str.replace_all(r'˹', '')
            expr = expr.str.replace_all(r'˺', '')
            
            expr = expr.str.replace_all('<gap>', '___GAP___', literal=True)
            expr = expr.str.replace_all(r'<([^>]+)>', r'${1}')
            expr = expr.str.replace_all(r'<|>', '')
            expr = expr.str.replace_all('___GAP___', '<gap>', literal=True)
            
            expr = expr.str.replace_all('(d)', '{d}', literal=True)
            expr = expr.str.replace_all('(ki)', '{ki}', literal=True)
            expr = expr.str.replace_all('(TÚG)', 'TÚG', literal=True)
            expr = expr.str.replace_all('Ḫ', 'H', literal=True)
            expr = expr.str.replace_all('ḫ', 'h', literal=True)
            expr = expr.str.replace_all('KÙ.B.', 'KÙ.BABBAR', literal=True)
            
            for dec, frac in self.fraction_map.items():
                expr = expr.str.replace_all(dec, frac, literal=True)
                
            expr = expr.str.replace_all(r'\s+', ' ').str.strip_chars()
            expr = pl.when(expr == "").then(pl.lit("<manual_split_needed>")).otherwise(expr)
            
            df = df.with_columns(expr.alias('transliteration'))

        if 'translation' in df.columns:
            expr = pl.col('translation').map_elements(
                lambda x: self._pre_process_python(x, True), return_dtype=pl.String
            )
            
            expr = expr.str.replace_all(r'<\((.*?)\)>', r'${1}')
            expr = expr.str.replace_all(r'!(sv|pr|sg|sn|eq|wp|cs)\b', '')
            expr = expr.str.replace_all(r'%(a|h|eg|e)\b', '')
            
            expr = expr.str.replace_all(r'(^|\s)[#"~|=^@](\s|$)', ' ')
            expr = expr.str.replace_all(r'(^|\s)[#"~|=^@](\s|$)', ' ')
            
            gap_patterns = r'\[x\]|\bx\b|…|\.\.\.|\(break\)|\(large break\)|\(n broken lines\)'
            expr = expr.str.replace_all(gap_patterns, '<gap>')
            expr = expr.str.replace_all('<big_gap>', '<gap>', literal=True)
            expr = expr.str.replace_all(r'(?:<gap>[\s\-]*)+<gap>', '<gap>')
            expr = expr.str.replace_all(r'(<gap>\s*)+', '<gap> ')
            
            expr = expr.str.replace_all('(?)', '', literal=True)
            expr = expr.str.replace_all(r'(?i)\bfem\.', '')
            expr = expr.str.replace_all(r'(?i)\bsing\.', '')
            expr = expr.str.replace_all(r'(?i)\bpl\.', '')
            expr = expr.str.replace_all(r'(?i)\bplural\b', '')
            expr = expr.str.replace_all(r'(?i)\bxx?\b', '')
            expr = expr.str.replace_all(r'\.\.+', '')
            expr = expr.str.replace_all(r'(^|\s)\?(\s|$)', ' ')
            expr = expr.str.replace_all(r'<<', '')
            expr = expr.str.replace_all(r'>>', '')
            expr = expr.str.replace_all(r'\[', '')
            expr = expr.str.replace_all(r'\]', '')
            expr = expr.str.replace_all(r'˹', '')
            expr = expr.str.replace_all(r'˺', '')
            
            expr = expr.str.replace_all('<gap>', '___GAP___', literal=True)
            expr = expr.str.replace_all(r'<([^>]+)>', r'${1}')
            expr = expr.str.replace_all(r'<|>', '')
            expr = expr.str.replace_all('___GAP___', '<gap>', literal=True)
            
            expr = expr.str.replace_all(r'(\b[a-zA-Z]+\b)\s*/\s*\b[a-zA-Z]+\b', r'${1}')
            
            expr = expr.str.replace_all('PN', '<gap>', literal=True)
            expr = expr.str.replace_all('-gold', 'pašallum gold', literal=True)
            expr = expr.str.replace_all('-tax', 'šadduātum tax', literal=True)
            expr = expr.str.replace_all('-textiles', 'kutānum textiles', literal=True)
            
            expr = expr.str.replace_all('1 / 12 (shekel)', '15 grains', literal=True)
            expr = expr.str.replace_all('5 / 12 shekel', '⅓ shekel 15 grains', literal=True)
            expr = expr.str.replace_all('5 11 / 12 shekels', '6 shekels less 15 grains', literal=True)
            expr = expr.str.replace_all('7 / 12 shekel', '½ shekel 15 grains', literal=True)
            
            expr = expr.str.replace_all('(d)', '{d}', literal=True)
            expr = expr.str.replace_all('(ki)', '{ki}', literal=True)
            expr = expr.str.replace_all('(TÚG)', 'TÚG', literal=True)
            
            for dec, frac in self.fraction_map.items():
                expr = expr.str.replace_all(dec, frac, literal=True)
                
            expr = expr.str.replace_all(r'\s+', ' ').str.strip_chars()
            expr = pl.when(expr == "").then(pl.lit("<manual_split_needed>")).otherwise(expr)
            
            df = df.with_columns(expr.alias('translation'))
            
        return df


class DatabaseCrossReferencer:
    def align_and_merge(self, primary_df: pl.DataFrame, secondary_df: pl.DataFrame, join_key: str) -> pl.DataFrame:
        if secondary_df is not None and not secondary_df.is_empty() and join_key in primary_df.columns and join_key in secondary_df.columns:
            aligned_df = primary_df.join(secondary_df, on=join_key, how="left")
            
            secondary_cols = [col for col in secondary_df.columns if col != join_key]
            fill_exprs = []
            
            for col in secondary_cols:
                dtype = aligned_df.schema[col]
                if dtype in [pl.String, pl.Categorical]:
                    fill_exprs.append(pl.col(col).fill_null("<manual_split_needed>"))
                else:
                    fill_exprs.append(pl.col(col).fill_null(-1))
            
            if fill_exprs:
                return aligned_df.with_columns(fill_exprs)
            return aligned_df
        return primary_df


class OCRDecryptor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_from_pdf(self) -> pl.DataFrame:
        return pl.DataFrame({
            "dictionary_lexicon": ["šadduātum", "kutānum"], 
            "meaning_class": ["tax", "textiles"]
        })
