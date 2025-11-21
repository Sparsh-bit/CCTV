from pathlib import Path

print('=' * 70)
print('FINAL VERIFICATION - CROWD BEHAVIOR FORECASTING SYSTEM')
print('=' * 70)

files = {
    'Trained Model': 'models/checkpoints/transformer_final.pt',
    'Test Data': 'data/raw/synthetic/sample.mp4',
    'Quick Inference Test': 'results/quick_inference_test.json',
    'Demo Inference Results': 'results/demo_inference_results.json',
    'Execution Summary': 'results/EXECUTION_SUMMARY.json',
    'Completion Report': 'COMPLETION_REPORT.md',
    'Deployment Summary': 'DEPLOYMENT_SUMMARY.md'
}

print('\nKEY FILES VERIFICATION:')
print('-' * 70)
all_exist = True
for name, path in files.items():
    exists = Path(path).exists()
    status = 'CHECKMARK' if exists else 'CROSS'
    print(f'{status} {name:30} {path}')
    if not exists:
        all_exist = False

print('-' * 70)
result = 'ALL FILES PRESENT' if all_exist else 'MISSING FILES'
print(f'\nOVERALL STATUS: {result}\n')
