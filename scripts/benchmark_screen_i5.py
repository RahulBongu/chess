#!/usr/bin/env python3
"""Milestone 5.4I.5 fixed screen benchmark.

Fixes the 5.4I.3 benchmark bugs:
1. orange_corners() fit axis-aligned extremes to non-board orange pixels.
   On dist/screen_nchw_clear.png (8368 orange px, largest contour area 246)
   it warped a tiny blob so every model predicted empty everywhere.
2. Board-level accuracy was labelled 'exact' while it counted per-square
   correct / 64 (full-board exact-match would be 0/64 boards exactly right
   whenever any square is wrong).
3. Digital-crop sanity used 2 crops/class and full-square resize to 64x64,
   while training resizes to 72x72 then centre-crops 64x64 (4:68).
4. Warp+80px crop+resize resampling differs from the device path
   (ChessSquareClassifier: bilinear sample at SQUARE_INSET for 64/72 ratio).

Corners now: largest orange-mask quad (approxPolyDP 4 pts) -> Canny outer
quad (square aspect only) -> explicit MANUAL fallback with overlay.
A MANUAL fallback is used for dist/screen_nchw_clear.png only: that frame
has no large orange overlay and its largest Canny quad is the phone UI
(33,53)-(1046,318, aspect 0.28), not the board. Manual corners reuse the
verified screen_assist_prompt.png orange quad (same propped camera, ~7 min
earlier/later) and are drawn to an overlay for human verification.
Uncertainty rule kept identical to old benchmark: conf<0.55 or margin<0.08.

Screen results only. Do not mix with physical 3D-board results.
Production model stays universal150 until held-out evidence says otherwise.

Usage:
  py -3 scripts/benchmark_screen_i5.py
  py -3 scripts/benchmark_screen_i5.py --models universal150_prod,wide150
  py -3 scripts/benchmark_screen_i5.py --json-out dist/other.json --check

Missing optional candidate files are skipped with a warning instead of
crashing, so a partial checkout still produces a comparable production
result. `--check` exits non-zero when a board needed the last-resort manual
fallback, which is the condition that silently corrupted the 5.4I.3 report.
"""
import argparse
import cv2, numpy as np, os, glob, json, hashlib
import onnxruntime as ort

CLASSES = ['empty','white_pawn','white_knight','white_bishop','white_rook','white_queen','white_king','black_pawn','black_knight','black_bishop','black_rook','black_queen','black_king']
START = [
    ['black_rook','black_knight','black_bishop','black_queen','black_king','black_bishop','black_knight','black_rook'],
    ['black_pawn']*8,
    ['empty']*8, ['empty']*8, ['empty']*8, ['empty']*8,
    ['white_pawn']*8,
    ['white_rook','white_knight','white_bishop','white_queen','white_king','white_bishop','white_knight','white_rook'],
]
AFTER_E4E5 = [
    ['black_rook','black_knight','black_bishop','black_queen','black_king','black_bishop','black_knight','black_rook'],
    ['black_pawn','black_pawn','black_pawn','black_pawn','empty','black_pawn','black_pawn','black_pawn'],
    ['empty']*8,
    ['empty','empty','empty','empty','black_pawn','empty','empty','empty'],
    ['empty','empty','empty','empty','white_pawn','empty','empty','empty'],
    ['empty']*8,
    ['white_pawn','white_pawn','white_pawn','white_pawn','empty','white_pawn','white_pawn','white_pawn'],
    ['white_rook','white_knight','white_bishop','white_queen','white_king','white_bishop','white_knight','white_rook'],
]

# Verified orange quad of dist/screen_assist_prompt.png (area 779291 contour).
# Reused ONLY for dist/screen_nchw_clear.png where auto detection fails.
MANUAL_E4E5_CORNERS = np.array([[104,730],[979,689],[970,1613],[117,1598]], dtype=np.float32)
# brown board quad from dist/i4_auto_13s.png orange contour (area 770647)
# is found automatically; constant is fallback if mask changes.
MANUAL_BROWN_CORNERS = np.array([[131,682],[1011,709],[996,1570],[130,1574]], dtype=np.float32)

TRAINING_CROP_RATIO = 64.0/72.0
SQUARE_INSET = (1.0 - TRAINING_CROP_RATIO)/2.0  # 0.0555..., matches Kotlin
UNCERT_CONF = 0.55
UNCERT_MARGIN = 0.08

MODELS = {
    'universal150_prod': 'models/rgb_square_classifier_v1_balanced_strong_universal150.onnx',
    'balanced150fix': 'models/rgb_square_classifier_v1_balanced_strong_balanced150fix.onnx',
    'zenodo_long': 'models/rgb_square_classifier_v1_zenodo_strong_long.onnx',
    'wide150': 'models/rgb_square_classifier_v1_zenodo_strong_wide150.onnx',
}

BOARDS = [
    ('green_start', 'dist/screen_assist_prompt.png', START),
    ('green_e4e5', 'dist/screen_nchw_clear.png', AFTER_E4E5),
    ('brown_start', 'dist/i4_auto_13s.png', START),
]

def order_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1,2)
    if len(pts)!=4:
        raise ValueError('need 4 points')
    s = pts[:,0]+pts[:,1]
    d = pts[:,0]-pts[:,1]
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]; bl = pts[np.argmin(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def orange_quad(img):
    orange = cv2.inRange(img, np.array([0,100,200]), np.array([60,200,255]))
    cnts,_ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    dbg = []
    for c in cnts[:3]:
        area = float(cv2.contourArea(c))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        dbg.append({'area':area,'npts':len(approx)})
        if len(approx)==4 and area>50000:
            return order_points(approx), dbg
    return None, dbg

def canny_quad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50,150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
    dbg = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)!=4:
            dbg.append({'area':area,'npts':len(approx)})
            continue
        pts = approx.reshape(4,2).astype(np.float32)
        xmin,xmax = pts[:,0].min(),pts[:,0].max()
        ymin,ymax = pts[:,1].min(),pts[:,1].max()
        w,h = xmax-xmin,ymax-ymin
        aspect = w/max(h,1e-6)
        dbg.append({'area':area,'npts':4,'w':float(w),'h':float(h),'aspect':float(aspect),'ymin':float(ymin)})
        if area>200000 and 0.8<aspect<1.25 and h>500 and ymin>300:
            return order_points(pts), dbg
    return None, dbg

def get_corners(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    q,dbg = orange_quad(img)
    if q is not None:
        return q,'orange_contour', {'orange_top':dbg}
    q,dbg2 = canny_quad(img)
    if q is not None:
        return q,'canny_quad', {'orange_top':dbg,'canny_top':dbg2}
    if 'screen_nchw_clear' in path:
        return MANUAL_E4E5_CORNERS.copy(),'manual_reused_from_start_orange', {'orange_top':dbg,'canny_top':dbg2,'note':'no large orange quad; Canny largest 4-pt is phone UI (aspect ~0.28) -> manual per audit overlay'}
    if 'i4_auto_13s' in path:
        return MANUAL_BROWN_CORNERS.copy(),'manual_brown', {'orange_top':dbg,'canny_top':dbg2}
    # Last resort (should not happen for current boards)
    return MANUAL_E4E5_CORNERS.copy(),'manual_fallback', {'orange_top':dbg,'canny_top':dbg2}

def save_overlay(path, corners, method):
    img = cv2.imread(path)
    draw = img.copy()
    pts = corners.astype(int)
    for i,p in enumerate(pts):
        cv2.circle(draw, tuple(p), 14, (0,255,0), -1)
        cv2.putText(draw, f'P{i}', tuple(p+np.array([10,-10])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    cv2.polylines(draw, [pts], True, (0,255,0), 3)
    os.makedirs('dist/audit_i5', exist_ok=True)
    out = f"dist/audit_i5/corners_i5_{os.path.basename(path).replace('.png','')}_{method}.png"
    cv2.imwrite(out, draw)
    return out

def stable_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    # logits may be (64,13) or (13,)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e/np.sum(e, axis=-1, keepdims=True)

def infer_batch(sess, nchw):
    # nchw: (N,3,64,64) float32 normalized (x/127.5-1)
    out = sess.run(None, {'input': nchw})[0]
    probs = stable_softmax(out)
    return out, probs

def sample_device_batch(img_bgr, corners):
    """Replicates ChessSquareClassifier classify() direct bilinear sampling."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h,w,_ = rgb.shape
    unit = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(unit, corners.astype(np.float32))  # unit->image
    a,b,c = float(H[0,0]),float(H[0,1]),float(H[0,2])
    d,e,f = float(H[1,0]),float(H[1,1]),float(H[1,2])
    g,hh = float(H[2,0]),float(H[2,1])
    batch = np.empty((64,3,64,64), dtype=np.float32)
    for r in range(8):
        for col in range(8):
            idx = r*8+col
            # Build uv grids
            xs = np.arange(64, dtype=np.float64)
            ys = np.arange(64, dtype=np.float64)
            u_1d = (col + SQUARE_INSET + (xs+0.5)/64.0*TRAINING_CROP_RATIO)/8.0
            v_1d = (r + SQUARE_INSET + (ys+0.5)/64.0*TRAINING_CROP_RATIO)/8.0
            U,V = np.meshgrid(u_1d, v_1d)  # (64,64) V rows, U cols
            den = g*U + hh*V + 1.0
            px = (a*U + b*V + c)/den
            py = (d*U + e*V + f)/den
            px = np.clip(px, 0, w-1); py = np.clip(py, 0, h-1)
            x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
            x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
            fx = (px-x0)[...,None]; fy = (py-y0)[...,None]
            # RGB 0-255 bilinear (matches sampleRgb)
            tl = rgb[y0,x0]; tr = rgb[y0,x1]; bl = rgb[y1,x0]; br = rgb[y1,x1]
            top = tl*(1-fx)+tr*fx
            bot = bl*(1-fx)+br*fx
            val = top*(1-fy)+bot*fy  # (64,64,3)
            norm = val/127.5 - 1.0   # (64,64,3)
            batch[idx] = np.transpose(norm,(2,0,1))
    return batch

def warp_batch(img_bgr, corners):
    dst = np.array([[0,0],[640,0],[640,640],[0,640]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    w = cv2.warpPerspective(img_bgr, M, (640,640))
    sq=80
    batch = np.empty((64,3,64,64), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            crop = w[r*sq:(r+1)*sq, c*sq:(c+1)*sq]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (64,64), interpolation=cv2.INTER_LINEAR)
            arr = (rgb.astype(np.float32)/127.5)-1.0
            batch[r*8+c] = np.transpose(arr,(2,0,1))
    return batch

def metrics_for(sess, batch, true_grid_flat):
    _, probs = infer_batch(sess, batch)
    preds_idx = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    top2 = np.sort(probs, axis=1)[:,-2:]
    margins = top2[:,1]-top2[:,0]
    uncertain = int(np.sum((confs<UNCERT_CONF)|(margins<UNCERT_MARGIN)))
    trues = np.array([CLASSES.index(t) for t in true_grid_flat])
    correct = int(np.sum(preds_idx==trues))
    occ = int(np.sum((preds_idx==0)==(trues==0)))
    occ_mask = trues!=0
    pred_occ = preds_idx!=0
    # occupied accuracy: exact piece among true occupied
    occupied_correct = int(np.sum(preds_idx[occ_mask]==trues[occ_mask])) if occ_mask.any() else 0
    occupied_n = int(np.sum(occ_mask))
    empty_correct = int(np.sum(preds_idx[~occ_mask]==0)) if (~occ_mask).any() else 0
    empty_n = int(np.sum(~occ_mask))
    colour_n=0; colour_c=0
    for ti,pi in zip(trues,preds_idx):
        if ti!=0 and pi!=0:
            colour_n+=1
            if CLASSES[ti].split('_')[0]==CLASSES[pi].split('_')[0]:
                colour_c+=1
    per = {}
    for true_idx in sorted(set(trues.tolist())):
        mask = trues==true_idx
        per[CLASSES[true_idx]] = f"{int(np.sum(preds_idx[mask]==true_idx))}/{int(np.sum(mask))}"
    return {
        'per_square_exact_n': correct,  # per-square count /64 (old 'exact')
        'per_square_exact_rate': round(correct/64,4),
        'board_exact': bool(correct==64),  # full-board exact-match (old label was misleading)
        'occ': occ,
        'occupied_exact': f'{occupied_correct}/{occupied_n}' if occupied_n else '0/0',
        'empty_exact': f'{empty_correct}/{empty_n}' if empty_n else '0/0',
        'colour': f'{colour_c}/{colour_n}' if colour_n else '0/0',
        'meanConf': round(float(np.mean(confs)),3),
        'uncertain': uncertain,
        'per_class_recall': per,
    }

def digital_eval(sess):
    files = sorted(glob.glob('dist/cache/digital_crops/*/*.png'))
    total=0; correct=0; occ=0; per={}
    all_probs=[]
    trues=[]
    # batch 256 for speed
    batch_imgs=[]; labels=[]
    def flush():
        nonlocal total,correct,occ
        if not batch_imgs:
            return
        arr = np.stack(batch_imgs,axis=0)  # (N,64,64,3) -> NCHW
        nchw = np.transpose(arr/127.5-1.0,(0,3,1,2)).astype(np.float32)
        _, probs = infer_batch(sess, nchw)
        preds = np.argmax(probs,axis=1)
        for li,p,pr in zip(labels,preds,probs):
            total+=1
            if int(p)==int(li): correct+=1
            if (int(p)==0)==(int(li)==0): occ+=1
            all_probs.append(pr); trues.append(li)
        batch_imgs.clear(); labels.clear()
    for f in files:
        cls_name = os.path.basename(os.path.dirname(f))
        if cls_name not in CLASSES:
            continue
        img = cv2.imread(f)
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 72x72 BGR->RGB
        # training eval: centre 64 (4:68)
        centre = rgb[4:68,4:68]
        batch_imgs.append(centre.astype(np.float32))
        labels.append(CLASSES.index(cls_name))
        if len(batch_imgs)>=256:
            flush()
    flush()
    trues=np.array(trues); preds=np.array([int(np.argmax(p)) for p in all_probs])
    per_class={}
    for ci,name in enumerate(CLASSES):
        m = trues==ci
        if m.sum():
            per_class[name]=f"{int(np.sum(preds[m]==ci))}/{int(np.sum(m))}"
    recalls=[float(v.split('/')[0])/float(v.split('/')[1]) for v in per_class.values()]
    return {
        'n': total,
        'per_square_exact_n': int(correct),
        'top1': round(correct/max(total,1),4),
        'macro_recall': round(float(np.mean(recalls)) if recalls else 0.0,4),
        'occ': int(occ),
        'per_class_recall': per_class,
        'preprocessing': '72x72 -> centre 64x64 (4:68) -> (x/127.5-1) NCHW, matches train_rgb_square_model.evaluate()',
    }

def sha256(p):
    import hashlib
    return hashlib.sha256(open(p,'rb').read()).hexdigest()

def resolve_models(selection):
    if selection in (None, '', 'all'):
        names = list(MODELS)
    else:
        names = [name.strip() for name in selection.split(',') if name.strip()]
    resolved = {}
    for name in names:
        if name not in MODELS:
            raise SystemExit(f"unknown model '{name}'; choose from {', '.join(MODELS)}")
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"WARNING skipping {name}: {path} not found", flush=True)
            continue
        resolved[name] = path
    if 'universal150_prod' not in resolved:
        raise SystemExit('production model universal150_prod is required')
    return resolved

def is_convex(pts):
    cross = []
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        cx, cy = pts[(i + 2) % 4]
        cross.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    return all(v > 0 for v in cross) or all(v < 0 for v in cross)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='all', help="comma list or 'all'")
    parser.add_argument('--json-out', default='dist/milestone_54i5_screen_benchmark.json')
    parser.add_argument('--check', action='store_true', help='exit 1 when a board used the last-resort fallback')
    args = parser.parse_args()

    model_paths = resolve_models(args.models)
    out_dir = os.path.dirname(args.json_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    corner_info={}
    for board_name,img_path,true_grid in BOARDS:
        if not os.path.exists(img_path):
            raise SystemExit(f"missing board image: {img_path}")
        corners,method,dbg = get_corners(img_path)
        overlay = save_overlay(img_path,corners,method)
        corner_info[board_name]={'path':img_path,'method':method,'corners':corners.tolist(),'debug':dbg,'overlay':overlay}
        print(board_name, method, corners.tolist(), overlay)

    results={}
    for name,path in model_paths.items():
        sess = ort.InferenceSession(path)
        entry={'onnx_sha256':sha256(path)}
        # device + warp per board
        for board_name,img_path,true_grid in BOARDS:
            img=cv2.imread(img_path)
            corners=np.array(corner_info[board_name]['corners'],dtype=np.float32)
            flat=[true_grid[r][c] for r in range(8) for c in range(8)]
            for mode,batchFn in [('device',lambda im: sample_device_batch(im,corners)),
                                  ('warp',lambda im: warp_batch(im,corners))]:
                batch=batchFn(img)
                entry[f'{board_name}_{mode}']=metrics_for(sess,batch,flat)
        entry['digital_crops']=digital_eval(sess)
        results[name]=entry
        print(name, {k:entry[k]['per_square_exact_n'] for k in entry if 'green' in k or 'brown' in k}, 'digital', entry['digital_crops']['top1'])

    # production asset check
    asset='modules/live-camera/android/src/main/assets/rgb_square_classifier_v1.onnx'
    prod='models/rgb_square_classifier_v1_balanced_strong_universal150.onnx'
    asset_sha=sha256(asset); prod_sha=sha256(prod)
    out={
        'date':'2026-11-06',
        'benchmark':'screen_i5_fixed',
        'supersedes':'dist/milestone_54i3_domain_benchmark.json (buggy orange_corners + per-square labelled exact)',
        'production_asset_sha256':asset_sha,
        'universal150_sha256':prod_sha,
        'asset_equals_universal150':bool(asset_sha==prod_sha),
        'corners':corner_info,
        'results':results,
        'notes':[
            'per_square_exact_n is per-square count /64; board_exact is full-board 64/64 match.',
            'device mode replicates ChessSquareClassifier bilinear+inset (64/72) direct sampling; warp mode is 640 warp +80px square+64 resize (old method, now with correct corners).',
            'digital_crops n=2600 (200/class), 72->centre64, not 2/class full-resize.',
            'green_e4e5 corners are MANUAL (auto failed); see overlays in dist/audit_i5/.',
            'brown_start assumes START position for dist/i4_auto_13s.png; check occupied metrics to confirm.',
            'Screen domain only; physical 3D-board results kept separate.',
        ]
    }
    with open(args.json_out,'w') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2)[:8000])

    problems = []
    for board_name, info in corner_info.items():
        if info['method'] == 'manual_fallback':
            problems.append(f"{board_name} used last-resort manual_fallback")
        if not is_convex(np.array(info['corners'], dtype=np.float32)):
            problems.append(f"{board_name} corners are not convex")
    if args.check:
        if problems:
            for problem in problems:
                print(f"CHECK FAIL {problem}")
            return 1
        print("CHECK OK all boards used a verified corner method")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
