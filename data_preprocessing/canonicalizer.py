import re
import random
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import tree_sitter_languages
from tree_sitter import Language, Parser

class UniversalCanonicalizer:
    def __init__(self):
        self.lang_map = {
            'python': 'python', 'py': 'python',
            'java': 'java',
            'cpp': 'cpp', 'c++': 'cpp', 'cxx': 'cpp',
            'c': 'c',
            'c#': 'c_sharp', 'csharp': 'c_sharp', 'cs': 'c_sharp',
            'javascript': 'javascript', 'js': 'javascript',
            'php': 'php',
            'go': 'go'
        }
        self.parsers = {}
        
    def __getstate__(self):
        """Called when Python tries to pickle this object for multiprocessing."""
        state = self.__dict__.copy()
        # Delete the un-picklable 'parsers' dictionary before sending to workers
        if 'parsers' in state:
            del state['parsers']
        return state

    def __setstate__(self, state):
        """Called when the worker receives the object and must re-initialize parsers."""
        self.__dict__.update(state)
        # Re-initialize the parsers dictionary
        self.parsers = {}
    
    def _get_parser(self, lang_key):
        """Lazy loader: Loads parser only if not already loaded in this process."""
        parser_name = self.lang_map.get(lang_key, 'python')
        
        # Check if we already have it in this process
        if parser_name in self.parsers:
            return self.parsers[parser_name]
        
        # If not, load it now
        try:
            # Note: tree-sitter_languages automatically handles the compiled shared object
            parser = tree_sitter_languages.get_parser(parser_name)
            self.parsers[parser_name] = parser
            return parser
        except Exception:
            return None

    def canonicalize(self, code_str, lang_label):
        if not isinstance(code_str, str): return ""
        
        # 1. Get Parser (Lazily)
        lang_key = str(lang_label).lower()
        parser = self._get_parser(lang_key)
        
        # Fallback
        if not parser: return code_str

        # 2. Parse
        code_bytes = code_str.encode('utf8')
        try:
            tree = parser.parse(code_bytes)
        except:
            return code_str
            
        # 3. Rename Identifiers
        code_bytes = self._rename_identifiers(code_bytes, tree.root_node)
        
        # 4. Decode
        try:
            canon_code = code_bytes.decode('utf8')
        except:
            return code_str 
            
        # 5. Flatten Layout
        parser_name = self.lang_map.get(lang_key, 'python')
        canon_code = self._flatten_layout(canon_code, parser_name)
        
        return canon_code

    def _rename_identifiers(self, code_bytes, root_node):
        replacements = [] 
        counters = {'VAR': 0, 'FUNC': 0, 'TYPE': 0}
        name_map = {} 
        
        keywords = {
            'if', 'else', 'for', 'while', 'return', 'class', 'def', 'void', 'int', 'float', 
            'double', 'string', 'public', 'private', 'static', 'import', 'from', 'include', 
            'package', 'namespace', 'struct', 'func', 'var', 'let', 'const', 'try', 'catch'
        }

        def traverse(node):
            if "identifier" in node.type or "variable" in node.type or "name" in node.type:
                if len(node.children) == 0:
                    original_name = code_bytes[node.start_byte : node.end_byte].decode('utf8', errors='ignore')
                    
                    if original_name not in keywords and len(original_name) > 1:
                        parent_type = node.parent.type if node.parent else ""
                        category = 'VAR'
                        
                        # Heuristic based on parent node type
                        if 'function' in parent_type or 'method' in parent_type or 'call' in parent_type:
                            category = 'FUNC'
                        elif 'class' in parent_type or 'type' in parent_type or 'struct' in parent_type:
                            category = 'TYPE'
                        
                        if original_name not in name_map:
                            name_map[original_name] = f"{category}_{counters[category]}"
                            counters[category] += 1
                        
                        new_name = name_map[original_name]
                        replacements.append((node.start_byte, node.end_byte, new_name))
            
            for child in node.children:
                traverse(child)

        traverse(root_node)
        
        # Apply replacements in reverse order of byte index
        replacements.sort(key=lambda x: x[0], reverse=True)
        mutable_code = bytearray(code_bytes)
        for start, end, new_text in replacements:
            mutable_code[start:end] = new_text.encode('utf8')
            
        return mutable_code

    def _flatten_layout(self, code_str, lang_type):
        lines = code_str.split('\n')
        processed_lines = []
        comment_pattern = re.compile(r'//.*|#.*') 
        for line in lines:
            line = re.sub(comment_pattern, '', line)
            if 'python' in lang_type:
                # Python: preserve leading indentation, remove tabs and trailing space
                line = line.replace('\t', '    ').rstrip()
            else:
                # Non-Python: remove all leading/trailing space
                line = line.strip()
            if line:
                processed_lines.append(line)
        return "\n".join(processed_lines)

class DualViewPipeline:
    def __init__(self, aug_prob=0.5):
        self.engine = UniversalCanonicalizer()
        self.aug_prob = aug_prob

    def process_batch(self, examples):
        """
        Processes a BATCH of data, adding the 'raw' view always and the 
        'structure' view probabilistically (aug_prob).
        """
        out_codes = []
        out_labels = []
        out_types = []
        
        batch_size = len(examples['code'])
        for i in range(batch_size):
            original_code = examples['code'][i]
            label = examples['label'][i]
            
            lang = examples['language'][i] if 'language' in examples and examples['language'][i] is not None else 'python'

            # 1. Always add raw view
            out_codes.append(original_code)
            out_labels.append(label)
            out_types.append('raw')

            # 2. Probabilistically add canonicalized view
            if random.random() < self.aug_prob:
                canon_code = self.engine.canonicalize(original_code, lang)
                
                # If canonicalization is successful
                if canon_code and len(canon_code) > 5:
                    out_codes.append(canon_code)
                    out_labels.append(label)
                    out_types.append('structure')
                else:
                    # Fallback: If parsing failed, append the original code as an additional raw view
                    out_codes.append(original_code)
                    out_labels.append(label)
                    out_types.append('raw')

        return {
            "code": out_codes, 
            "label": out_labels, 
            "view_type": out_types
        }

def build_and_save_dataset(input_parquet_path, output_disk_path):
    print(f"Starting Pipeline on {input_parquet_path}")
    
    df = pd.read_parquet(input_parquet_path)
    df = df.dropna(subset=['code', 'label'])
    dataset = Dataset.from_pandas(df)
    
    pipeline = DualViewPipeline(aug_prob=0.6) 
    
    print("   -> Processing... (This creates the augmented rows)")
    
    augmented_dataset = dataset.map(
        pipeline.process_batch, 
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=dataset.column_names
    )
    
    augmented_dataset = augmented_dataset.shuffle(seed=42)
    
    print(f"   -> Saving to {output_disk_path}...")
    augmented_dataset.save_to_disk(output_disk_path)
    
    print("\nDATASET AUGMENTATION COMPLETE")
    print(f"   Original Rows: {len(dataset)}")
    print(f"   Augmented Rows: {len(augmented_dataset)}")
    
    return augmented_dataset

def prepare_tagged_data(train_path, json_path):
    print("1. Loading and Tagging Dataset...")
    
    df = pd.read_parquet(train_path).dropna(subset=['code', 'label']).reset_index(drop=True)
    
    if os.path.exists(json_path):
        print(f"   -> Loading Hard Negatives from {json_path}")
        with open(json_path, "r") as f:
            indices_data = json.load(f)
        hard_indices_set = set(indices_data["hard_indices"])
        
        df['is_hard'] = df.index.isin(hard_indices_set)
        print(f"   -> Tagged {df['is_hard'].sum()} samples as HARD.")
    else:
        print("   ⚠️ Hard Negative JSON not found. Treating all as Easy (flat augmentation).")
        df['is_hard'] = False

    return df

class EasyHardDualViewPipeline:
    def __init__(self, aug_prob_hard=1.0, aug_prob_easy=0.3):
        self.engine = UniversalCanonicalizer() 
        self.aug_prob_hard = aug_prob_hard
        self.aug_prob_easy = aug_prob_easy

    def process_batch(self, examples):
        """
        Processes a BATCH of data with Dynamic Probabilities based on 'is_hard' tag.
        """
        out_codes = []
        out_labels = []
        out_types = []
        
        batch_size = len(examples['code'])
        rng_values = np.random.random(batch_size)
        
        for i in range(batch_size):
            original_code = examples['code'][i]
            label = examples['label'][i]
            is_hard = examples['is_hard'][i]
            
            lang = examples['language'][i] if 'language' in examples and examples['language'][i] else 'python'

            # 1. Keep the raw view
            out_codes.append(original_code)
            out_labels.append(label)
            out_types.append('raw')

            # 2. Determine threshold and probabilistically generate structural view
            threshold = self.aug_prob_hard if is_hard else self.aug_prob_easy
            
            if rng_values[i] < threshold:
                canon_code = self.engine.canonicalize(original_code, lang)
                
                # Only append structural view if parsing was successful
                if canon_code and len(canon_code) > 5:
                    out_codes.append(canon_code)
                    out_labels.append(label)
                    out_types.append('structure')

        return {
            "code": out_codes, 
            "label": out_labels, 
            "view_type": out_types
        }
    
def run_phase_2_pipeline(train_parquet_path, hard_json_path, output_parquet_path):
    # 1. Prepare Dataframe with 'is_hard' tags
    df = prepare_tagged_data(train_parquet_path, hard_json_path)
    dataset = Dataset.from_pandas(df)
    
    # 2. Initialize the Pipeline
    pipeline = EasyHardDualViewPipeline(aug_prob_hard=1.0, aug_prob_easy=0.3)
    
    print(f"Starting Augmentation (Batch Size: 1000, Cores: 4)...")
    
    # 3. Apply Map with Batching
    augmented_dataset = dataset.map(
        pipeline.process_batch, 
        batched=True,
        batch_size=1000, 
        num_proc=4,
        remove_columns=dataset.column_names 
    )
    
    # 4. Shuffle
    print("3. Shuffling...")
    augmented_dataset = augmented_dataset.shuffle(seed=42)
    
    print(f"4. Saving to {output_parquet_path}...")
    
    df_export = augmented_dataset.to_pandas()
    df_export.to_parquet(output_parquet_path, engine='pyarrow', index=False)
    
    print("PHASE 2 COMPLETE")
    print(f"   Original Size: {len(dataset)}")
    print(f"   Augmented Size: {len(augmented_dataset)}")
    
def run_tta_inference(
    test_data_path: str,
    model_dir: str,
    base_model_name: str = "microsoft/unixcoder-base",
    raw_weight: float = 0.2,
    struct_weight: float = 0.8
) -> pd.DataFrame:
    """
    Performs Test-Time Augmentation (TTA) inference by ensembling 
    predictions from the raw code view and the canonical structure view.
    
    Weights should sum to 1.0 (e.g., raw_weight=0.2, struct_weight=0.8)
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load Model & Tokenizer
    print(f"Loading Tokenizer from base ({base_model_name})...")
    tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading Model Weights from {model_dir}...")
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
    except:
        model = RobertaForSequenceClassification.from_pretrained(model_dir, use_safetensors=False)
        
    model.to(device)
    model.eval()

    try:
        # Initialize the canonicalizer for on-the-fly TTA
        canonicalizer_engine = UniversalCanonicalizer() 
    except NameError:
        raise NameError("UniversalCanonicalizer class not defined.")
        
    test_df = pd.read_parquet(test_data_path)
    final_preds = []

    print(f"Starting TTA Inference on {len(test_df)} samples. Weights: Raw={raw_weight}, Structure={struct_weight}")

    # 2. TTA Inference Loop
    for i in tqdm(range(len(test_df))):
        code = test_df.iloc[i]['code']
        lang = test_df.iloc[i]['language'] if 'language' in test_df.columns else 'python'
        
        # A. Raw View
        inputs_raw = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits_raw = model(**inputs_raw).logits
            probs_raw = F.softmax(logits_raw, dim=-1)
            
        # B. Structural View
        canon_code = canonicalizer_engine.canonicalize(code, lang)
        
        # Fallback if canonicalization fails
        if len(canon_code.strip()) < 5:
            canon_code = code
            
        inputs_struct = tokenizer(canon_code, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits_struct = model(**inputs_struct).logits
            probs_struct = F.softmax(logits_struct, dim=-1)
            
        # C. Ensemble
        avg_probs = (raw_weight * probs_raw) + (struct_weight * probs_struct)
        pred_idx = torch.argmax(avg_probs).item()
        final_preds.append(pred_idx)
    
    # 3. Create Submission DataFrame
    submission = pd.DataFrame()
    if 'ID' in test_df.columns:
        submission['ID'] = test_df['ID']
    elif 'id' in test_df.columns:
        submission['ID'] = test_df['id']
    else:
        submission['ID'] = test_df.index

    submission['label'] = final_preds
    
    return submission

def run_test():
    
    print("\n\n--- Running on dummy data ---")
    
    # 1. Create dummy multi-lingual data
    debug_data = pd.DataFrame([
        {"code": "def calc_sum(a, b):\n    # A comment\n    return a + b", "label": 1, "language": "python"},
        {"code": "int main() {\n  int x = 10;\n  return x;\n}", "label": 2, "language": "cpp"},
        {"code": "public class Test {\n  public void run() {\n    System.out.println(\"Hi\");\n  }\n}", "label": 3, "language": "java"}
    ])
    
    # 2. Initialize Pipeline (with 100% augmentation probability for testing)
    pipeline = DualViewPipeline(aug_prob=1.0)
    
    for i, row in debug_data.iterrows():
        
        single_example_batch = {
            'code': [row['code']],
            'label': [row['label']],
            'language': [row['language']]
        }
        
        print(f"\n--- Language: {row['language']} ---")
        print("Original:\n", row['code'])
        
        # Process the properly formatted batch
        out = pipeline.process_batch(single_example_batch)
        
        # Check the Structural View (index 1 of the output lists)
        struct_code = out['code'][1]
        print("Canonicalized:\n", struct_code)
        
        # Assertions/Checks (similar to original cell)
        if "VAR_" in struct_code or "FUNC_" in struct_code:
            print("✅ Renaming Worked")
        if "//" not in struct_code and "#" not in struct_code:
            print("✅ Comments Stripped")

if __name__ == "__main__":
    run_test()