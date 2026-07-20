import pandas as pd
import cv2, os

df = pd.read_csv('results/viffd_features_hybrid.csv')
vdir = 'data/viffd'
for _, r in df.head(5).iterrows():
    vn = r['video_name']
    vp = os.path.join(vdir, vn)
    print(f"{vn}: exists={os.path.exists(vp)}")

# Check if we can get n_frames from the videos
print("\nChecking video lengths...")
lengths = []
for _, r in df.iterrows():
    vn = r['video_name']
    vp = os.path.join(vdir, vn)
    if os.path.exists(vp):
        cap = cv2.VideoCapture(vp)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        lengths.append(nf)
    else:
        lengths.append(-1)

df['n_frames'] = lengths
valid = df[df['n_frames'] > 0]
print(f"Videos found: {len(valid)}/{len(df)}")
print(f"n_frames stats:\n{valid['n_frames'].describe()}")
print(f"\nBy label:")
print(valid.groupby('label')['n_frames'].describe())

# Save augmented CSV
df.to_csv('results/viffd_features_hybrid_with_nframes.csv', index=False)
print("\nSaved: results/viffd_features_hybrid_with_nframes.csv")
