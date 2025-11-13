from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
from pathlib import Path
import re
# from utils.helper import Helpers
# from multiprocessing import Pool

r"""
```c
FILE -> src\data\results\grammar
NEXT:
grammar\page_1.txt -> src\data\contents\page_1_clear.txt
```
"""

# ⚙️ GPU config
torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 🔧 Tải model/tokenizer một lần duy nhất
# model_name = "Qwen/Qwen3-4B-Instruct-2507"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Gensyn/Qwen2.5-0.5B-Instruct"

# 🚀 CÁC TÙY CHỌN CẢI THIỆN HIỆU SUẤT:
# 1. Thử mô hình lớn hơn: "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"
# 2. Thử mô hình chuyên biệt tiếng Việt: "vinai/PhoGPT-7B5-Instruct" (nếu có)

# 📂 Đọc file .txt đầu vào
def read_all_txt_list(folder: str) -> dict[int, str]:
    files = sorted([f for f in os.listdir(folder) if f.endswith("clear.txt") and "_ocr" not in f], 
                   key=lambda x: int(x.split('_')[1].split('.')[0]))
    return {int(f.split('_')[1].split('.')[0]): open(os.path.join(folder, f), encoding="utf-8").read() 
            for f in files}

import os
import re
from typing import List

def extract_unique_words(raw_folder: str) -> List[str]:
    """
    Đọc tất cả file .txt trong folder và trích xuất tập hợp các từ duy nhất.
    
    Args:
        raw_folder (str): Đường dẫn đến folder chứa các file .txt
        
    Returns:
        Set[str]: Tập hợp các từ duy nhất (chỉ chứa chữ cái tiếng Việt)
    """
    char_counter = set()
    
    for filename in sorted(os.listdir(raw_folder)):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            words = content.split()
            
            # Loại bỏ ký tự không phải chữ cái
            words = [re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9]', '', word) for word in words]
            
            # Loại bỏ ký tự đặc biệt và số, chỉ giữ từ chứa chữ cái
            words = [word for word in words if word.isalpha() and not word.isdigit()]
            
            # Thêm vào tập hợp
            char_counter = char_counter.union(set(words))
    
    return list(char_counter)

unique_words = extract_unique_words(r"src\data\raw")

def tokenize_words(text: str) -> list:
    """Tách văn bản thành danh sách các từ (loại bỏ dấu câu và ký tự đặc biệt)"""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    return words

def create_word_dictionary(text: str) -> set:
    """
    Tạo từ điển (set) các từ duy nhất từ văn bản gốc
    """
    words = tokenize_words(text)
    return set(words)

def detect_english_words(text: str) -> list:
    """
    Phát hiện các từ tiếng Anh nguyên bản rõ ràng (loại bỏ danh từ quốc tế)
    
    Returns:
        list: Danh sách các từ tiếng Anh tìm thấy (chỉ những từ nguyên bản tiếng Anh)
    """
    # Danh sách các từ tiếng Anh NGUYÊN BẢN - các từ đơn giản, động từ, giới từ, liên từ
    # Loại bỏ các danh từ quốc tế thường dùng trong tiếng Việt (email, website, etc.)
    pure_english_words = {
        # Articles, pronouns
        'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them'
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'am',
        'have', 'has', 'does', 'did',
        'will', 'would', 'should', 'could', 'might', 'must', 'can',
        'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went',
        'said', 'see', 'saw', 'know', 'knew', 'think', 'thought',
        # Common conjunctions, prepositions
        'and', 'or', 'but', 'yet',
        'on', 'at', 'for', 'by', 'from', 'with', 'as',
        'about', 'after', 'before', 'between', 'during', 'through',
        'into', 'over', 'under', 'above', 'below', 'off', 'out', 'up', 'down',
        # Common adverbs, quantifiers
        'not', 'yes', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'only', 'same', 'just', 'very', 'how', 'why',
        'where', 'when', 'what', 'which', 'who', 'whom', 'whose', 'get', 'got', 'getting', 'make', 'made', 'making', 'take', 'took', 'taking',
        'come', 'came', 'coming', 'go', 'went', 'going', 'said', 'saying',
        'see', 'saw', 'seeing', 'know', 'knew', 'knowing', 'think', 'thought', 'thinking',
        'give', 'gave', 'giving', 'find', 'found', 'finding', 'tell', 'told', 'telling',
        'ask', 'asked', 'asking', 'work', 'worked', 'working', 'call', 'called', 'calling',
        'try', 'tried', 'trying', 'use', 'used', 'using', 'feel', 'felt', 'feeling',
        'become', 'became', 'becoming', 'leave', 'left', 'leaving', 'put', 'putting',
        'mean', 'meant', 'meaning', 'keep', 'kept', 'keeping', 'help', 'helped', 'helping',
        'talk', 'talked', 'talking', 'turn', 'turned', 'turning', 'start', 'started', 'starting',
        'show', 'showed', 'showing', 'hear', 'heard', 'hearing', 'let', 'letting', 'hold', 'held', 'holding',
        'bring', 'brought', 'bringing', 'begin', 'began', 'beginning', 'seem', 'seemed', 'seeming',
        'write', 'wrote', 'writing', 'written', 'provide', 'provided', 'providing',
        'play', 'played', 'playing', 'run', 'ran', 'running', 'move', 'moved', 'moving',
        'like', 'liked', 'liking', 'live', 'lived', 'living', 'believe', 'believed', 'believing',
        'want', 'wanted', 'wanting', 'look', 'looked', 'looking', 'appear', 'appeared', 'appearing',
        'watch', 'watched', 'watching', 'follow', 'followed', 'following', 'stop', 'stopped', 'stopping',
        'create', 'created', 'creating', 'speak', 'spoke', 'speaking', 'read', 'reading',
        'allow', 'allowed', 'allowing', 'add', 'added', 'adding', 'spend', 'spent', 'spending',
        'grow', 'grew', 'growing', 'grown', 'draw', 'drew', 'drawing', 'drawn', 'break', 'broke', 'breaking', 'broken',
        'happen', 'happened', 'happening', 'choose', 'chose', 'choosing', 'chosen', 'deal', 'dealt', 'dealing',
        'serve', 'served', 'serving', 'eat', 'ate', 'eating', 'eaten', 'cover', 'covered', 'covering',
        'catch', 'caught', 'catching', 'draw', 'draw', 'drive', 'drove', 'driving', 'driven',
        'die', 'died', 'dying', 'face', 'faced', 'facing', 'fail', 'failed', 'failing',
        'gain', 'gained', 'gaining', 'hang', 'hung', 'hanging', 'hit', 'hitting',
        'hold', 'hole', 'hunt', 'hunted', 'hunting', 'include', 'included', 'including',
        'increase', 'increased', 'increasing', 'involve', 'involved', 'involving', 'join', 'joined', 'joining',
        'jump', 'jumped', 'jumping', 'kill', 'killed', 'killing', 'laid', 'laying',
        'lead', 'led', 'leading', 'learn', 'learned', 'learning', 'learnt', 'leave', 'left',
        'light', 'lit', 'lighting', 'lighted', 'listen', 'listened', 'listening', 'lose', 'lost', 'losing',
        'love', 'loved', 'loving', 'measure', 'measured', 'measuring', 'meet', 'met', 'meeting',
        'mind', 'minded', 'minding', 'miss', 'missed', 'missing', 'obtain', 'obtained', 'obtaining',
        'occur', 'occurred', 'occurring', 'offer', 'offered', 'offering', 'open', 'opened', 'opening',
        'order', 'ordered', 'ordering', 'own', 'owned', 'owning', 'paint', 'painted', 'painting',
        'pass', 'passed', 'passing', 'pay', 'paid', 'paying', 'perform', 'performed', 'performing',
        'perhaps', 'pick', 'picked', 'picking', 'point', 'pointed', 'pointing', 'prepare', 'prepared', 'preparing',
        'present', 'presented', 'presenting', 'prevent', 'prevented', 'preventing', 'print', 'printed', 'printing',
        'promise', 'promised', 'promising', 'prove', 'proved', 'proving', 'proven', 'pull', 'pulled', 'pulling',
        'push', 'pushed', 'pushing', 'raise', 'raised', 'raising', 'reach', 'reached', 'reaching',
        'realize', 'realized', 'realizing', 'receive', 'received', 'receiving', 'record', 'recorded', 'recording',
        'reduce', 'reduced', 'reducing', 'refuse', 'refused', 'refusing', 'regard', 'regarded', 'regarding',
        'remember', 'remembered', 'remembering', 'remove', 'removed', 'removing', 'repeat', 'repeated', 'repeating',
        'replace', 'replaced', 'replacing', 'report', 'reported', 'reporting', 'require', 'required', 'requiring',
        'result', 'resulted', 'resulting', 'return', 'returned', 'returning', 'reveal', 'revealed', 'revealing',
        'review', 'reviewed', 'reviewing', 'ride', 'rode', 'riding', 'ridden', 'ring', 'rang', 'ringing', 'rung',
        'rise', 'rose', 'rising', 'risen', 'risk', 'risked', 'risking', 'roll', 'rolled', 'rolling',
        'rub', 'rubbed', 'rubbing', 'rule', 'ruled', 'ruling', 'rush', 'rushed', 'rushing',
        'sail', 'sailed', 'sailing', 'satisfy', 'satisfied', 'satisfying', 'save', 'saved', 'saving',
        'search', 'searched', 'searching', 'seat', 'seated', 'seating', 'secure', 'secured', 'securing',
        'seek', 'sought', 'seeking', 'seem', 'seemed', 'seeming', 'seize', 'seized', 'seizing',
        'sell', 'sold', 'selling', 'send', 'sent', 'sending', 'sense', 'sensed', 'sensing',
        'separate', 'separated', 'separating', 'set', 'setting', 'settle', 'settled', 'settling',
        'shake', 'shook', 'shaking', 'shaken', 'share', 'shared', 'sharing', 'shift', 'shifted', 'shifting',
        'shine', 'shone', 'shining', 'shined', 'ship', 'shipped', 'shipping', 'shoot', 'shot', 'shooting',
        'shop', 'shopped', 'shopping', 'shut', 'shutting', 'sight', 'sighted', 'sighting',
        'sign', 'signed', 'signing', 'signal', 'signaled', 'signaling', 'sing', 'singing',
        'sink', 'sank', 'sinking', 'sunken', 'sit', 'sat', 'sitting', 'size', 'sized', 'sizing',
        'sketch', 'sketched', 'sketching', 'sleep', 'slept', 'sleeping', 'slide', 'slid', 'sliding',
        'smile', 'smiled', 'smiling', 'smoke', 'smoked', 'smoking', 'smooth', 'smoothed', 'smoothing',
        'snow', 'snowed', 'snowing', 'solve', 'solved', 'solving', 'sort', 'sorted', 'sorting',
        'sound', 'sounded', 'sounding', 'speak', 'spoke', 'speaking', 'spoken', 'speed', 'sped', 'speeding',
        'spend', 'spent', 'spending', 'spell', 'spelled', 'spelling', 'spelt', 'split', 'splitting',
        'spread', 'spreading', 'spring', 'sprung', 'springing', 'stand', 'stood', 'standing',
        'stare', 'stared', 'staring', 'start', 'started', 'starting', 'state', 'stated', 'stating',
        'stay', 'stayed', 'staying', 'steal', 'stole', 'stealing', 'stolen', 'step', 'stepped', 'stepping',
        'stick', 'stuck', 'sticking', 'still', 'stilled', 'stilling', 'sting', 'stung', 'stinging',
        'stink', 'stank', 'stinking', 'stunk', 'stitch', 'stitched', 'stitching', 'stop', 'stopped', 'stopping',
        'store', 'stored', 'storing', 'storm', 'stormed', 'storming', 'strain', 'strained', 'straining',
        'stream', 'streamed', 'streaming', 'street', 'stretch', 'stretched', 'stretching', 'strike', 'struck', 'striking', 'struck',
        'string', 'strung', 'stringing', 'strip', 'stripped', 'stripping', 'stroke', 'stroked', 'stroking',
        'struggle', 'struggled', 'struggling', 'study', 'studied', 'studying', 'stuff', 'stuffed', 'stuffing',
        'stumble', 'stumbled', 'stumbling', 'submit', 'submitted', 'submitting', 'succeed', 'succeeded', 'succeeding',
        'suck', 'sucked', 'sucking', 'suffer', 'suffered', 'suffering', 'suggest', 'suggested', 'suggesting',
        'suit', 'suited', 'suiting', 'sum', 'summed', 'summing', 'supply', 'supplied', 'supplying',
        'support', 'supported', 'supporting', 'suppose', 'supposed', 'supposing', 'suppress', 'suppressed', 'suppressing',
        'sure', 'surely', 'surface', 'surfaced', 'surfacing', 'surge', 'surged', 'surging',
        'surprise', 'surprised', 'surprising', 'surround', 'surrounded', 'surrounding', 'survey', 'surveyed', 'surveying',
        'survive', 'survived', 'surviving', 'suspect', 'suspected', 'suspecting', 'suspend', 'suspended', 'suspending',
        'sustain', 'sustained', 'sustaining', 'swallow', 'swallowed', 'swallowing', 'swear', 'swore', 'swearing', 'sworn',
        'sweat', 'sweated', 'sweating', 'sweep', 'swept', 'sweeping', 'swell', 'swelled', 'swelling', 'swollen',
        'swim', 'swam', 'swimming', 'swum', 'swing', 'swung', 'swinging', 'switch', 'switched', 'switching',
        'swoop', 'swooped', 'swooping', 'symbol', 'sympathize', 'sympathized', 'sympathizing', 'symptom',
        'sync', 'synced', 'syncing', 'system', 'systematize', 'systematized', 'systematizing', 'table', 'tabled', 'tabling',
        'tackle', 'tackled', 'tackling', 'tag', 'tagged', 'tagging', 'tail', 'tailed', 'tailing',
        'take', 'took', 'taking', 'taken', 'tale', 'talk', 'talked', 'talking', 'tally', 'tallied', 'tallying',
        'tame', 'tamed', 'taming', 'tan', 'tanned', 'tanning', 'tangle', 'tangled', 'tangling',
        'tap', 'tapped', 'tapping', 'tape', 'taped', 'taping', 'target', 'targeted', 'targeting',
        'task', 'tasked', 'tasking', 'taste', 'tasted', 'tasting', 'tattoo', 'tattooed', 'tattooing',
        'teach', 'taught', 'teaching', 'tease', 'teased', 'teasing', 'telephone', 'telephoned', 'telephoning',
        'tell', 'told', 'telling', 'temper', 'tempered', 'tempering', 'tempt', 'tempted', 'tempting',
        'tend', 'tended', 'tending', 'tender', 'tendered', 'tendering', 'tense', 'tensed', 'tensing',
        'term', 'termed', 'terming', 'terrify', 'terrified', 'terrifying', 'test', 'tested', 'testing',
        'text', 'texted', 'texting', 'thank', 'thanked', 'thanking', 'thaw', 'thawed', 'thawing',
        'theater', 'theft', 'theme', 'theory', 'therapy', 'there', 'therefore', 'thermal',
        'think', 'thought', 'thinking', 'thin', 'thinned', 'thinning', 'thirst', 'thirsted', 'thirsting',
        'thorn', 'thorough', 'those', 'thread', 'threaded', 'threading', 'threat', 'threatened', 'threatening',
        'three', 'thresh', 'threshed', 'threshing', 'threshold', 'threw', 'thrice', 'thrift',
        'thrill', 'thrilled', 'thrilling', 'thrive', 'thrived', 'thriving', 'throve', 'throb', 'throbbed', 'throbbing',
        'throne', 'throng', 'thronged', 'thronging', 'throttle', 'throttled', 'throttling', 'through', 'throw', 'threw', 'throwing', 'thrown',
        'thrust', 'thrusting', 'thumb', 'thumbed', 'thumbing', 'thump', 'thumped', 'thumping',
        'thunder', 'thundered', 'thundering', 'thus', 'thwart', 'thwarted', 'thwarting', 'ticket', 'ticketed', 'ticketing',
        'tickle', 'tickled', 'tickling', 'tide', 'tided', 'tiding', 'tidy', 'tidied', 'tidying',
        'tie', 'tied', 'tying', 'tier', 'tiered', 'tiering', 'tiger', 'tight', 'tighten', 'tightened', 'tightening',
        'tights', 'tile', 'tiled', 'tiling', 'till', 'tilled', 'tilling', 'tilt', 'tilted', 'tilting',
        'timber', 'time', 'timed', 'timing', 'timid', 'tined', 'tinfoil', 'tinge', 'tinged', 'tingeing', 'tingle', 'tingled', 'tingling',
        'tinker', 'tinkered', 'tinkering', 'tint', 'tinted', 'tinting', 'tiny', 'tip', 'tipped', 'tipping',
        'tipsy', 'tire', 'tired', 'tiring', 'tissue', 'titan', 'titanic', 'tithe', 'tithed', 'tithing',
        'title', 'titled', 'titling', 'titter', 'tittered', 'tittering', 'toad', 'toast', 'toasted', 'toasting',
        'tobacco', 'today', 'toddle', 'toddled', 'toddling', 'toe', 'toed', 'toeing', 'toffee',
        'tofu', 'together', 'toggle', 'toggled', 'toggling', 'toil', 'toiled', 'toiling', 'token',
        'tolerate', 'tolerated', 'tolerating', 'toll', 'tolled', 'tolling', 'tomato', 'tomb', 'tombed', 'tombing',
        'tombstone', 'tomcat', 'tome', 'tomorrow', 'ton', 'tone', 'toned', 'toning', 'tongs',
        'tongue', 'tonic', 'tonight', 'tonnage', 'tonsil', 'too', 'took', 'tool', 'tooled', 'tooling',
        'toot', 'tooted', 'tooting', 'tooth', 'toothbrush', 'toothpaste', 'toothpick', 'toots', 'top', 'topped', 'topping',
        'topic', 'topical', 'topography', 'topple', 'toppled', 'toppling', 'topsy', 'torch', 'torched', 'torching',
        'tore', 'torment', 'tormented', 'tormenting', 'torn', 'tornado', 'torpedo', 'torpedoed', 'torpedoing',
        'torpedo', 'torpor', 'torque', 'torrent', 'torrid', 'torso', 'tort', 'tortoise', 'torture', 'tortured', 'torturing',
        'torus', 'toss', 'tossed', 'tossing', 'tot', 'total', 'totaled', 'totaling', 'totalitarian',
        'totality', 'totality', 'tote', 'toted', 'toting', 'totem', 'totter', 'tottered', 'tottering',
        'toucan', 'touch', 'touched', 'touching', 'touchy', 'tough', 'toughen', 'toughened', 'toughening',
        'toughness', 'tour', 'toured', 'touring', 'tourism', 'tourist', 'tournament', 'tousle', 'tousled', 'tousling',
        'tout', 'touted', 'touting', 'tow', 'towed', 'towing', 'towage', 'toward', 'towards',
        'towel', 'toweled', 'toweling', 'towelled', 'towelling', 'tower', 'towered', 'towering',
        'town', 'township', 'toxic', 'toxin', 'toy', 'toyed', 'toying', 'trace', 'traced', 'tracing',
        'tracer', 'trachea', 'track', 'tracked', 'tracking', 'tract', 'traction', 'tractor',
        'trade', 'traded', 'trading', 'trader', 'tradition', 'traditional', 'traffic', 'trafficked', 'trafficking',
        'trafficker', 'tragedy', 'tragic', 'tragically', 'trail', 'trailed', 'trailing', 'trailer',
        'train', 'trained', 'training', 'trainee', 'trainer', 'traipse', 'traipsed', 'traipsing',
        'trait', 'traitor', 'traitorous', 'trajectory', 'tram', 'tramcar', 'tramp', 'tramped', 'tramping',
        'trample', 'trampled', 'trampling', 'trampoline', 'trampolined', 'trampolining', 'trance', 'tranquil',
        'tranquilizer', 'tranquilize', 'tranquilized', 'tranquillizing', 'tranquillity', 'transact', 'transacted', 'transacting',
        'transaction', 'transcend', 'transcended', 'transcending', 'transcendent', 'transcendental', 'transcendence',
        'transcontinental', 'transcribe', 'transcribed', 'transcribing', 'transcript', 'transcription', 'transcriber',
        'transducer', 'transeunt', 'transept', 'transfer', 'transferred', 'transferring', 'transferable', 'transference',
        'transfiguration', 'transfigure', 'transfigured', 'transfiguring', 'transfix', 'transfixed', 'transfixing',
        'transform', 'transformed', 'transforming', 'transformation', 'transformer', 'transfusion', 'transgress',
        'transgressed', 'transgressing', 'transgression', 'transgressor', 'tranship', 'transshipped', 'transhipping',
        'transhumance', 'transience', 'transient', 'transistor', 'transit', 'transited', 'transiting',
        'transition', 'transitioned', 'transitioning', 'transitional', 'transitive', 'transitivity', 'transitory',
        'translatable', 'translate', 'translated', 'translating', 'translation', 'translator', 'transliterate',
        'transliterated', 'transliterating', 'transliteration', 'translucence', 'translucency', 'translucent',
        'transmigration', 'transmigrate', 'transmigrated', 'transmigrating', 'transmissible', 'transmission',
        'transmit', 'transmitted', 'transmitting', 'transmitter', 'transmittal', 'transmittable', 'transmogrification',
        'transmogrify', 'transmogrified', 'transmogrifying', 'transmontane', 'transmutation', 'transmute',
        'transmuted', 'transmuting', 'transnational', 'transoceanic', 'transom', 'transparency', 'transparent',
        'transpicuous', 'transpiration', 'transpire', 'transpired', 'transpiring', 'transplant', 'transplanted', 'transplanting',
        'transplantation', 'transplanter', 'transplant', 'transplanter', 'transpolar', 'transpontine', 'transport',
        'transported', 'transporting', 'transportable', 'transportation', 'transporter', 'transpontine', 'transpose',
        'transposed', 'transposing', 'transposition', 'transposal', 'transposable', 'transylvania',
        'transylvanian', 'truncyear', 'transferals', 'tansy', 'trans'
    }
    
    words = tokenize_words(text)
    english_words = [w for w in words if w in pure_english_words]
    return english_words

def check_vocabulary_match(original: str, corrected: str) -> dict:
    """
    Kiểm tra xem tất cả các từ sau xử lý có nằm trong từ điển của văn bản gốc không
    Chỉ tính chữ cái, không tính số
    KHÔNG PHÂN BIỆT CHỮ HOA CHỮ THƯỜNG
    
    Returns:
        dict: {
            'original_vocab_size': int,
            'corrected_vocab_size': int,
            'new_words_count': int,
            'new_words': list,
            'all_words_in_dict': bool,
            'english_words': list
        }
    """
    # Chuyển tất cả về lowercase để so sánh
    original_dict = create_word_dictionary(original.lower())
    corrected_words = tokenize_words(corrected.lower())
    corrected_dict = set(corrected_words)
    
    # Tìm các từ mới (không có trong từ điển gốc)
    new_words = corrected_dict - original_dict
    
    # Chỉ tính chữ cái, loại bỏ số
    new_words_alpha_only = [word for word in new_words if word.isalpha()]
    
    # QUAN TRỌNG: Lọc các từ mới hợp lệ
    # 1. Loại bỏ ký tự đơn lẻ nếu là phần của từ viết tắt: "ESG" → "E.S.G" → "e","s","g"
    # 2. Cho phép mở rộng từ viết tắt: "CP" → "Cổ Phần" (nếu C và P nằm trong đầu các từ mới)
    original_lower = original.lower()
    new_words_filtered = []
    
    # Tìm các từ viết tắt trong original (từ có 2-5 ký tự, toàn chữ hoa trong bản gốc)
    acronyms_in_original = set()
    for orig_word in tokenize_words(original):  # Không lowercase ở đây
        if 2 <= len(orig_word) <= 5 and orig_word.isupper():
            acronyms_in_original.add(orig_word.lower())
    
    for word in new_words_alpha_only:
        if len(word) == 1:
            # Ký tự đơn lẻ - kiểm tra xem có phải là phần của từ viết tắt không
            is_part_of_acronym = any(word in orig_word for orig_word in original_dict if len(orig_word) > 1)
            if not is_part_of_acronym:
                new_words_filtered.append(word)
        else:
            # Từ có 2+ ký tự - kiểm tra xem có phải là phần mở rộng của từ viết tắt không
            # Ví dụ: "cổ", "phần" có thể là mở rộng của "cp" (cổ phần)
            is_expansion_of_acronym = False
            for acronym in acronyms_in_original:
                # Lấy chữ cái đầu của các từ mới
                first_letters = ''.join([w[0] for w in new_words_alpha_only if len(w) > 1])
                if acronym in first_letters or first_letters.startswith(acronym):
                    is_expansion_of_acronym = True
                    break
            
            if not is_expansion_of_acronym:
                new_words_filtered.append(word)
    
    new_words_alpha_only = new_words_filtered
    
    # Phát hiện từ tiếng Anh
    english_words = detect_english_words(corrected)
    
    return {
        'original_vocab_size': len(original_dict),
        'corrected_vocab_size': len(corrected_dict),
        'new_words_count': len(new_words_alpha_only),  # Chỉ đếm chữ cái
        'new_words': sorted(list(new_words_alpha_only)),  # Chỉ trả về chữ cái
        'all_words_in_dict': len(new_words_alpha_only) == 0,  # Chỉ kiểm tra chữ cái
        'english_words': english_words
    }

def compare_texts(original: str, corrected: str) -> dict:
    """
    So sánh 2 văn bản và tìm các từ khác nhau sử dụng difflib
    KHÔNG PHÂN BIỆT CHỮ HOA CHỮ THƯỜNG
    
    Returns:
        dict: Thông tin chi tiết về sự khác biệt
    """
    from difflib import SequenceMatcher
    
    # Chuyển tất cả về lowercase để so sánh
    original_words = tokenize_words(original.lower())
    corrected_words = tokenize_words(corrected.lower())
    
    # Sử dụng SequenceMatcher để tìm sự khác biệt
    matcher = SequenceMatcher(None, original_words, corrected_words)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Từ bị thay thế
            for idx in range(max(i2-i1, j2-j1)):
                orig = original_words[i1+idx] if i1+idx < i2 else None
                corr = corrected_words[j1+idx] if j1+idx < j2 else None
                if orig and corr:
                    differences.append(('replace', orig, corr, i1+idx+1))
                elif orig:
                    differences.append(('delete', orig, '', i1+idx+1))
                elif corr:
                    differences.append(('insert', '', corr, i1+idx+1))
        elif tag == 'delete':
            # Từ bị xóa
            for idx in range(i1, i2):
                differences.append(('delete', original_words[idx], '', idx+1))
        elif tag == 'insert':
            # Từ được thêm vào
            for idx in range(j1, j2):
                differences.append(('insert', '', corrected_words[idx], i1+1))
    
    return {
        'total_words_original': len(original_words),
        'total_words_corrected': len(corrected_words),
        'different_words_count': len(differences),
        'differences': differences
    }

# helpers = Helpers()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
input_folder = os.path.join(project_root, "src", "data", "contents")
output_folder = os.path.join(project_root, "src", "data", "grammar")

# Note: We now read files directly in the main loop, so no need for pages dictionary
# pages = read_all_txt_list(input_folder)
# pages = [' '.join(helpers.bm25_preprocessing_func(page)) for page in pages]
# pages = [page.replace("Báo cáo phát triển bền vững", " ") for page in pages if page]

os.makedirs(output_folder, exist_ok=True)

# Xác định thư mục gốc của dự án
project_root = Path(__file__).resolve().parent.parent.parent

# Thư mục chứa file contents
contents_dir = project_root / "src" / "data" / "contents"

# Thư mục output
output_folder = project_root / "src" / "data" / "grammar"

# VỊ TRÍ BẮT ĐẦU XỬ LÝ
START_PAGE = 1  # Thay đổi số này để bắt đầu từ trang khác

print(model_name)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,          # Sử dụng bfloat16 để tăng tốc và tiết kiệm VRAM
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # ⚡️ Tăng tốc attention
    ).eval()                                 # Chỉ dùng cho inference
    model = torch.compile(model, mode="reduce-overhead")             # Tăng tốc nếu PyTorch >= 2.0
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
except Exception as e:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,          # Sử dụng bfloat16 để tăng tốc và tiết kiệm VRAM
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # ⚡️ Tăng tốc attention
    ).eval()                                 # Chỉ dùng cho inference
    model = torch.compile(model, mode="reduce-overhead")             # Tăng tốc nếu PyTorch >= 2.0
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 🧠 Hàm hiệu đính chính tả
def correct_vietnamese(text: str, repeat_reminder: int, model, tokenizer, use_enhanced_prompt: bool = False, memory_examples: list | None = None) -> str:
    """
    Hiệu đính chính tả tiếng Việt với khả năng học từ các ví dụ trước đó
    
    Args:
        text: Văn bản cần hiệu đính
        repeat_reminder: Số lần retry (không sử dụng trong logic hiện tại)
        model: Model LLM
        tokenizer: Tokenizer
        use_enhanced_prompt: Nếu True, sử dụng prompt nâng cao với định dạng chi tiết hơn
        memory_examples: List of (original, corrected, similarity) tuples - Các ví dụ tốt nhất để model tham khảo
    
    Returns:
        str: Văn bản đã được hiệu đính
    """
    prompt = f"\n\n{text}"
    
    # Prompt cơ bản - đơn giản, ít ràng buộc
    basic_prompt = """You are an **intelligent Vietnamese spelling and grammar corrector and formatter**.

**Your tasks**:

* Check and **correct all Vietnamese spelling, punctuation, and grammar errors** with perfect linguistic accuracy.
* **Preserve the original meaning, structure, and formatting** — do **not** summarize, interpret, or add/remove content.
* If the text contains **multiple semantic blocks**, separate them using the delimiter `\\n\\n`.
* If the text includes **table-like or list-like data**, present it clearly as a **flat ordered list**, in this format:
  Item 1: [Tên] – [Mô tả + dữ liệu chính]
  Item 2: [Tên] – [Mô tả + dữ liệu chính]
* If the text includes **labels, headers, or hierarchical sections** (ví dụ: "Đặc biệt trọng yếu", "Rất trọng yếu", "Trọng yếu", ...), **preserve them exactly as input** and keep their associated items grouped below.
* If the text describes a **chart, process, or workflow**, express it as a **logical sequence** using arrows (>>) between steps.
* Maintain correct **Vietnamese capitalization, punctuation, spacing, and diacritics**.
* Preserve all **proper nouns, abbreviations, organizational names, and references** exactly (e.g. E.S.G, CP, CT, ESG, CK, BIWASE, ...).
* Ensure **consistent line breaks and spacing** as in the input, especially for headings, sections, and chapter titles.
* **No English response**, **Just Vietnamese**.

**Output requirements:**

* Output **only the corrected text**, with the same formatting, line breaks, and structure as the input.
* Do **not** include explanations, comments, or any extra symbols (no markdown, no bullets unless already in text).
* The output must look like a polished, publication-ready Vietnamese document while retaining the original structure and flow.
"""
    
    # Prompt nâng cao - chi tiết hơn, định dạng rõ ràng
    enhanced_prompt = """You are an **intelligent Vietnamese spelling and grammar corrector**.

**Your tasks**:

* Check and **correct all Vietnamese spelling, punctuation, and grammar errors**.
* **Preserve the original meaning, character count, and formatting** — do not add, remove, or summarize content.
* If the text contains **multiple semantic blocks**, separate them using the delimiter `\\n\\n`.
* If the text includes **table-like or list-like data**, present it clearly as a **flat ordered list** (không chia cột, không nhóm).
  Ví dụ:

  1. [Tên chủ đề] – [Mô tả/ngữ cảnh nếu có]
  2. [Tên chủ đề] – [Mô tả/ngữ cảnh nếu có]
* If the text includes **labels or axes** (ví dụ: "Rất trọng yếu", "Trọng yếu", ...), hãy giữ nguyên chúng như tiêu đề dòng, không sắp xếp lại.
* If the text describes a **chart, process, or workflow**, trình bày lại thành **trình tự logic** bằng mũi tên (>>) giữa các bước.
* Maintain correct **Vietnamese capitalization, punctuation, and spacing**.
* Do not change proper nouns, organizational names, or abbreviations (ví dụ: E.S.G, CP, CT, ESG, CK, ...).
* **No English response**, **Just Vietnamese**.

**Output requirements:**

* Chỉ xuất ra **văn bản đã được chỉnh chính tả và ngữ pháp**, không có giải thích, không thêm ký hiệu định dạng.
* **Giữ nguyên bố cục như đầu vào** (bao gồm tiêu đề, danh sách, mục lục chương, và cấu trúc phân cấp).
"""
    
    # Chọn prompt phù hợp
    system_prompt = enhanced_prompt if use_enhanced_prompt else basic_prompt
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    
    # 💾 THÊM BỘ NHỚ: Nếu có ví dụ tốt, thêm vào messages để model học
    if memory_examples and len(memory_examples) > 0:
        print(f"  🧠 Sử dụng {len(memory_examples)} ví dụ tốt nhất làm reference")
        for idx, (orig_text, corrected_text, sim_score) in enumerate(memory_examples, 1):
            messages.append({
                "role": "user",
                "content": f"Ví dụ {idx} (similarity: {sim_score:.3f}):\n\n{orig_text}"
            })
            messages.append({
                "role": "assistant",
                "content": corrected_text
            })
    
    # Thêm văn bản hiện tại cần xử lý
    messages.append({"role": "user", "content": prompt})

    total_len = len(text.split())
    print(f"Total input tokens: {total_len}")
    print(f"📝 Using {'enhanced' if use_enhanced_prompt else 'basic'} prompt")

    max_new_tokens = 2048
        
    inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inp], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out_ids[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return result

def extract_page_number(filename):
    """Trích xuất số trang từ tên file.

    Hỗ trợ các dạng:
    - "page_1" -> 1
    - "page_cleared_2" -> 2
    - Bất kỳ tên nào kết thúc bằng số -> lấy nhóm số cuối cùng
    """
    m = re.search(r"(\d+)$", filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

# Thu thập tất cả file text từ contents (không có suffix _ocr)
content_files = {}
for content_file in contents_dir.glob("page_cleared_*.txt"):
    # Bỏ qua các file có suffix _ocr
    if "_ocr" not in content_file.stem:
        page_num = extract_page_number(content_file.stem)
        if page_num is not None and page_num >= START_PAGE:
            content_files[page_num] = content_file

# Thu thập tất cả file từ output_folder (grammar)
grammar_files = {}
for grammar_file in output_folder.glob("page_cleared_*.txt"):
    page_num = extract_page_number(grammar_file.stem)
    if page_num is not None and page_num >= START_PAGE:
        grammar_files[page_num] = grammar_file

# Hiển thị kết quả
print(f"Tìm thấy {len(content_files)} file contents (từ trang {START_PAGE})")
print(f"Tìm thấy {len(grammar_files)} file grammar (từ trang {START_PAGE})")

# Add project root to path for imports
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# from src.models.embedd import QwenEmbedding
from src.models.halong_embedd import HalongEmbedding as QwenEmbedding

embedding = QwenEmbedding()
array_similarity = []
array_word_changes = []

# 💾 BỘ NHỚ: Lưu 2 phản hồi tốt nhất để model tham khảo
best_examples = []  # List of (original, corrected, similarity) tuples

# So khớp các file theo page number
for page_num in sorted(content_files.keys()):
    content_path = content_files[page_num]
    grammar_path = grammar_files.get(page_num)
    
    # Ensure grammar_path exists
    if grammar_path is None:
        # Đồng bộ với pattern đọc "page_cleared_*.txt"
        grammar_path = output_folder / f"page_cleared_{page_num}.txt"
    
    print(f"\n{'='*70}")
    print(f"📄 Page {page_num}:")
    print(f"{'='*70}")
    print(f"  Content: {content_path}")
    print(f"  Grammar: {grammar_path}")
    
    # Read content from the actual file
    with open(content_path, 'r', encoding='utf-8') as f:
        page_content = f.read()
    
    # LẦN 1: Thử với basic prompt (+ memory nếu có)
    corrected_text = correct_vietnamese(
        page_content, 
        page_num, 
        model, 
        tokenizer, 
        use_enhanced_prompt=False,
        memory_examples=best_examples  # Truyền bộ nhớ vào
    )
    similarity = embedding.calculate_similarity(page_content, corrected_text)
    
    if similarity < 0.8 and len(page_content.split()) < 20:
        print(f"⚠️ Trang {page_num}: Similarity thấp ({similarity:.3f}) và văn bản ngắn (<20 từ). Bỏ qua trang này.")
        with open(grammar_path, 'w', encoding='utf-8') as f:
            f.write(page_content)
        continue

    # Kiểm tra từ điển
    vocab_check = check_vocabulary_match(page_content, corrected_text)
    
    # ĐIỀU KIỆN RETRY: similarity < 0.95 HOẶC có nhiều từ mới (> 5 từ)
    max_retry = 1  # Chỉ retry 1 lần với enhanced prompt
    retry_count = 0
    
    if (similarity < 0.95 or vocab_check['new_words_count'] > 5) and retry_count < max_retry:
        print(f"\n🔄 RETRY với enhanced prompt (similarity={similarity:.3f}, new_words={vocab_check['new_words_count']})")
        retry_count += 1
        
        # LẦN 2: Thử với enhanced prompt (+ memory)
        corrected_text = correct_vietnamese(
            page_content, 
            page_num, 
            model, 
            tokenizer, 
            use_enhanced_prompt=True,
            memory_examples=best_examples  # Truyền bộ nhớ vào
        )
        similarity = embedding.calculate_similarity(page_content, corrected_text)
        vocab_check = check_vocabulary_match(page_content, corrected_text)
        
        print(f"✅ Sau retry: similarity={similarity:.3f}, new_words={vocab_check['new_words_count']}")
    
    # 💾 CẬP NHẬT BỘ NHỚ: Lưu các ví dụ tốt nhất
    # Chỉ lưu nếu similarity >= 0.95 và ít từ mới
    if similarity >= 0.95 and vocab_check['new_words_count'] <= 3:
        best_examples.append((page_content, corrected_text, similarity))
        # Sắp xếp theo similarity giảm dần và chỉ giữ 2 ví dụ tốt nhất
        best_examples.sort(key=lambda x: x[2], reverse=True)
        best_examples = best_examples[:2]
        print(f"  💾 Đã lưu vào bộ nhớ (tổng: {len(best_examples)} ví dụ, similarity: {similarity:.3f})")
    
    # So sánh chi tiết
    comparison = compare_texts(page_content, corrected_text)
    
    print(f"\n📊 Kết quả cuối cùng:")
    print(f"  - Similarity: {similarity:.4f}")
    print(f"  - Số từ gốc: {comparison['total_words_original']}")
    print(f"  - Số từ đã sửa: {comparison['total_words_corrected']}")
    print(f"  - Số từ khác nhau: {comparison['different_words_count']}")
    print(f"  - Từ mới (chữ cái): {vocab_check['new_words_count']} từ")
    
    if vocab_check['new_words_count'] > 0:
        print(f"  - Các từ mới: {', '.join(vocab_check['new_words'][:10])}")
    
    if vocab_check['english_words']:
        print(f"  - ⚠️ Phát hiện {len(vocab_check['english_words'])} từ tiếng Anh: {', '.join(vocab_check['english_words'][:5])}")
    
    if comparison['different_words_count'] > 0:
        print(f"\n📝 Chi tiết thay đổi:")
        for diff in comparison['differences'][:10]:
            print(f"    • {diff}")
    
    # Lưu kết quả
    with open(grammar_path, 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    
    print(f"\n✅ Đã lưu kết quả vào: {grammar_path}")
    
    array_similarity.append({
        'page': page_num,
        'similarity': similarity,
        'retry_count': retry_count
    })
    
    array_word_changes.append({
        'page': page_num,
        'word_diff': comparison['different_words_count'],
        'new_words': vocab_check['new_words_count']
    })

# Tổng kết
print(f"\n{'='*70}")
print(f"📈 TỔNG KẾT:")
print(f"{'='*70}")
print(f"Tổng số trang đã xử lý: {len(array_similarity)}")
avg_similarity = sum([x['similarity'] for x in array_similarity]) / len(array_similarity) if array_similarity else 0
print(f"Similarity trung bình: {avg_similarity:.4f}")
retry_pages = [x for x in array_similarity if x['retry_count'] > 0]
print(f"Số trang cần retry: {len(retry_pages)}")

if retry_pages:
    print(f"\nCác trang đã retry:")
    for page_info in retry_pages:
        print(f"  - Page {page_info['page']}: similarity={page_info['similarity']:.3f}")
