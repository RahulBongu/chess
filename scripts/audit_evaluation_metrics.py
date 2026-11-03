import json

def main():
    r = json.load(open('dist/rgb-square-model-report-balanced_strong_balanced150fix.json', encoding='utf-8'))
    val = r['validation']
    f = val['per_source']['fenit:val']
    z = val['per_source']['zenodo:valid:set4']

    print('=== FENiT Validation (40 real phone photographs, 2560 squares) ===')
    print(f"Overall Top-1: {f['top1']*100:.2f}% (N = {f['n']})")
    print(f"Overall Top-2: {f['top2']*100:.2f}% (N = {f['n']})")
    print(f"Empty-square accuracy: {f['empty_top1']*100:.2f}% (N = 2054)")
    print(f"Occupied-piece accuracy: {f['occupied_top1']*100:.2f}% (N = 506)")
    print(f"Complete-board exact matches: {f['board_exact']*100:.2f}% ({int(round(f['board_exact']*f['board_count']))}/{f['board_count']})")
    print(f"Macro recall: {f['macro_recall']*100:.2f}% (13 classes)")
    print('\nPer-class accuracy on real FENiT photos:')
    for c, d in f['per_class'].items():
        print(f"  {c:14s}: {d['recall']*100:5.1f}% (N = {d['n']})")

    print('\n=== Zenodo Set 4 Holdout (832 crops, physical Set 4) ===')
    print(f"Overall Top-1: {z['top1']*100:.2f}% (N = {z['n']})")
    print(f"Overall Top-2: {z['top2']*100:.2f}% (N = {z['n']})")
    print(f"Empty recall: {z['per_class']['empty']['recall']*100:.1f}% (N = 64)")
    occupied_correct = sum(z['per_class'][c]['recall'] * 64 for c in z['per_class'] if c != 'empty')
    print(f"Occupied recall: {occupied_correct / 768 * 100:.1f}% (N = 768)")
    for c, d in z['per_class'].items():
        print(f"  {c:14s}: {d['recall']*100:5.1f}% (N = {d['n']})")

if __name__ == '__main__':
    main()
