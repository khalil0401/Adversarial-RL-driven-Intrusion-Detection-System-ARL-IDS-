import json

# Load notebook
with open('kaggle_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the evaluation cell and add seed parameter
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell.get('source', []))
        if 'eval_args = argparse.Namespace(' in source_text and 'evaluate(eval_args)' in source_text:
            # Add seed parameter to eval_args
            for line_idx, line in enumerate(cell['source']):
                if '    encoder_path="results/checkpoints/encoder.pth"' in line:
                    # Add seed parameter after encoder_path
                    cell['source'].insert(line_idx + 1, ',\n')
                    cell['source'].insert(line_idx + 2, '    seed=42\n')
                    print(f"Added seed parameter to evaluation cell {cell_idx}")
                    break
            break

# Save
with open('kaggle_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("OK - Fixed evaluation seed parameter!")
