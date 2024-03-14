from datasets import load_dataset, DatasetDict, Dataset
import os
import random

langs_mteb = ['sqi-eng', 'fry-eng', 'kur-eng', 'tur-eng', 'deu-eng', 'nld-eng', 'ron-eng', 'ang-eng', 'ido-eng', 'jav-eng', 'isl-eng', 'slv-eng', 'cym-eng', 'kaz-eng', 'est-eng', 'heb-eng', 'gla-eng', 'mar-eng', 'lat-eng', 'bel-eng', 'pms-eng', 'gle-eng', 'pes-eng', 'nob-eng', 'bul-eng', 'cbk-eng', 'hun-eng', 'uig-eng', 'rus-eng', 'spa-eng', 'hye-eng', 'tel-eng', 'afr-eng', 'mon-eng', 'arz-eng', 'hrv-eng', 'nov-eng', 'gsw-eng', 'nds-eng', 'ukr-eng', 'uzb-eng', 'lit-eng', 'ina-eng', 'lfn-eng', 'zsm-eng', 'ita-eng', 'cmn-eng', 'lvs-eng', 'glg-eng', 'ceb-eng', 'bre-eng', 'ben-eng', 'swg-eng', 'arq-eng', 'kab-eng', 'fra-eng', 'por-eng', 'tat-eng', 'oci-eng', 'pol-eng', 'war-eng', 'aze-eng', 'vie-eng', 'nno-eng', 'cha-eng', 'mhr-eng', 'dan-eng', 'ell-eng', 'amh-eng', 'pam-eng', 'hsb-eng', 'srp-eng', 'epo-eng', 'kzj-eng', 'awa-eng', 'fao-eng', 'mal-eng', 'ile-eng', 'bos-eng', 'cor-eng', 'cat-eng', 'eus-eng', 'yue-eng', 'swe-eng', 'dtp-eng', 'kat-eng', 'jpn-eng', 'csb-eng', 'xho-eng', 'orv-eng', 'ind-eng', 'tuk-eng', 'max-eng', 'swh-eng', 'hin-eng', 'dsb-eng', 'ber-eng', 'tam-eng', 'slk-eng', 'tgl-eng', 'ast-eng', 'mkd-eng', 'khm-eng', 'ces-eng', 'tzl-eng', 'urd-eng', 'ara-eng', 'kor-eng', 'yid-eng', 'fin-eng', 'tha-eng', 'wuu-eng']

iso3to2 = language_mapping = {
    "sqi": "sq", "fry": "fy", "kur": "ku", "tur": "tr", "deu": "de", "nld": "nl", "ron": "ro", "ang": "None",
    "ido": "io", "jav": "jv", "isl": "is", "slv": "sl", "cym": "cy", "kaz": "kk", "est": "et", "heb": "he",
    "gla": "gd", "mar": "mr", "lat": "la", "bel": "be", "pms": "None", "gle": "ga", "pes": "fa", "nob": "nb",
    "bul": "bg", "cbk": "None", "hun": "hu", "uig": "ug", "rus": "ru", "spa": "es", "hye": "hy", "tel": "te",
    "afr": "af", "mon": "mn", "arz": "None", "hrv": "hr", "nov": "None", "gsw": "None", "nds": "None",
    "ukr": "uk", "uzb": "uz", "lit": "lt", "ina": "ia", "lfn": "None", "zsm": "None", "ita": "it", "cmn": "zh",
    "lvs": "lv", "glg": "gl", "ceb": "None", "bre": "br", "ben": "bn", "swg": "None", "arq": "None",
    "kab": "None", "fra": "fr", "por": "pt", "tat": "tt", "oci": "oc", "pol": "pl", "war": "None",
    "aze": "az", "vie": "vi", "nno": "None", "cha": "None", "mhr": "None", "dan": "da", "ell": "el",
    "amh": "am", "pam": "None", "hsb": "None", "srp": "sr", "epo": "eo", "kzj": "None", "awa": "None",
    "fao": "fo", "mal": "ml", "ile": "None", "bos": "bs", "cor": "kw", "cat": "ca", "eus": "eu", "yue": "None",
    "swe": "sv", "dtp": "None", "kat": "ka", "jpn": "ja", "csb": "None", "xho": "None", "orv": "None",
    "ind": "id", "tuk": "tk", "max": "None", "swh": "None", "hin": "hi", "dsb": "None", "ber": "None",
    "tam": "ta", "slk": "sk", "tgl": "tl", "ast": "None", "mkd": "mk", "khm": "km", "ces": "cs", "tzl": "None",
    "urd": "ur", "ara": "ar", "kor": "ko", "yid": "yi", "fin": "fi", "tha": "th", "wuu": "None"
}

converted_languages = [language_mapping[code] if code in language_mapping else "None" for code in [
    "sqi", "fry", "kur", "tur", "deu", "nld", "ron", "ang", "ido", "jav", "isl", "slv", "cym", "kaz", "est", "heb",
    "gla", "mar", "lat", "bel", "pms", "gle", "pes", "nob", "bul", "cbk", "hun", "uig", "rus", "spa", "hye", "tel",
    "afr", "mon", "arz", "hrv", "nov", "gsw", "nds", "ukr", "uzb", "lit", "ina", "lfn", "zsm", "ita", "cmn", "lvs",
    "glg", "ceb", "bre", "ben", "swg", "arq", "kab", "fra", "por", "tat", "oci", "pol", "war", "aze", "vie", "nno",
    "cha", "mhr", "dan", "ell", "amh", "pam", "hsb", "srp", "epo", "kzj", "awa", "fao", "mal", "ile", "bos", "cor",
    "cat", "eus", "yue", "swe", "dtp", "kat", "jpn", "csb", "xho", "orv", "ind", "tuk", "max", "swh", "hin", "dsb",
    "ber", "tam", "slk", "tgl", "ast", "mkd", "khm", "ces", "tzl", "urd", "ara", "kor", "yid", "fin", "tha", "wuu"
]]

langs_ours = ["af", "am", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "du", "ha", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lo", "lt", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "om", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "xh", "yi", "zhs", "zht"]
iso2to3 = {v: k for k, v in iso3to2.items()}

def data_load():
    dataset_train = DatasetDict()
    dataset_valid = DatasetDict()

    for (root, dirs, files) in os.walk("./content/Tatoeba_txt/"):
        for pairs in files:
            langs = pairs.split(".")[0].split("_")
            lang1, lang2 = langs[0], langs[1]
            print(pairs, lang1 + "-" + lang2)
            result = tatoeba_filtering(lang1, lang2, root + pairs)
            if result:
                train, valid = result
                if len(train) != 0:
                    os.makedirs("./content/Tatoeba/"+lang1 + "-" + lang2 + "/")
                    dataset_train[lang1 + "-" + lang2] = Dataset.from_list(train)
                    dataset_train[lang1 + "-" + lang2].to_parquet("./content/Tatoeba/"+lang1 + "-" + lang2+"/train.parquet")
                if len(valid) != 0:
                    os.makedirs("./content/Tatoeba/"+lang1 + "-" + lang2 + "/", exist_ok=True)
                    dataset_valid[lang1 + "-" + lang2] = Dataset.from_list(valid)
                    dataset_valid[lang1 + "-" + lang2].to_parquet("./content/Tatoeba/"+lang1 + "-" + lang2+"/valid.parquet")
    
    dataset_valid.push_to_hub("./wecover/tatoeba_valid/")


def tatoeba_filtering(lang1, lang2, file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    lines = [line.lstrip('\t') for line in lines]
    lines = [line.rstrip('\t') for line in lines]
    if len(lines) == 0:
        return

    data = {}

    pair = None
    if lang1 == "en":
        if lang2 in iso2to3:
            pair = iso2to3[lang2] + "-eng"
    elif lang2 == "en":
        if lang1 in iso2to3:
            pair = iso2to3[lang1] + "-eng"
    
    if pair:
        dataset = load_dataset("mteb/tatoeba-bitext-mining", pair)
        for elem in dataset["test"]:
            sentence1 = elem['sentence1'].rstrip('\n')
            sentence2 = elem['sentence2'].rstrip('\n')
            if not sentence1[-1].isalpha():
                sentence1 = sentence1[:-1]
            if not sentence2[-1].isalpha():
                sentence2 = sentence2[:-1]  
            if not sentence1[0].isalpha():
                sentence1 = sentence1[1:]
            if not sentence2[0].isalpha():
                sentence2 = sentence2[1:] 
            data[sentence2] = sentence1
        print("mteb exists")
    else:
        print("no mteb")

    result = []
    cnt = 0
    for line in lines:
        if '\t' in line:
            strings = line.split('\t')
            dict = {'sentence1':strings[0], 'sentence2':strings[1], 'guid': cnt, 'lang1':lang1, 'lang2':lang2}
            for key, elem in data.items():
                if key in strings[0] or elem in strings[1]:
                    break
            else:
                cnt+=1
                result.append(dict)

    print(len(lines), "->", len(result))
    random.shuffle(result)

    train_len = int(len(result) * 0.8) if int(len(result) * 0.8) < 20000 else 20000
    test_len = int(len(result) * 0.2) if int(len(result) * 0.2) < 1000 else 1000
    return result[:train_len], result[train_len:train_len + test_len]

if __name__ == "__main__":
    data_load()