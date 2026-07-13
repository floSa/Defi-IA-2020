import csv, math, sys
from collections import Counter

path = sys.argv[1]
csv.field_size_limit(10**9)

n=0; n_test=0; n_train=0
ups_sum=0.0; ups_min=math.inf; ups_max=-math.inf
ups_eq1=0; ups_le0=0; ups_neg=0
hist=Counter()          # coarse ups histogram (train)
authors=set(); links=set(); subs=Counter()
body_len_sum=0; body_len_max=0; empty_body=0
deleted_author=0
sample_ups=[]           # reservoir-ish: keep every 400th for quantiles

with open(path, 'r', encoding='utf-8', newline='') as f:
    r = csv.reader(f)
    header = next(r)
    idx = {c:i for i,c in enumerate(header)}
    iu, ia, il, isb, ib = idx['ups'], idx['author'], idx['link_id'], idx['subreddit'], idx['body']
    for row in r:
        n += 1
        ups = row[iu].strip()
        subs[row[isb]] += 1
        links.add(row[il])
        au = row[ia]
        authors.add(au)
        if au in ('[deleted]',''):
            deleted_author += 1
        bl = len(row[ib])
        body_len_sum += bl
        if bl > body_len_max: body_len_max = bl
        if bl == 0: empty_body += 1
        if ups == '' or ups.lower()=='nan':
            n_test += 1
        else:
            v = float(ups)
            n_train += 1
            ups_sum += v
            ups_min = min(ups_min, v); ups_max = max(ups_max, v)
            if v==1: ups_eq1 += 1
            if v<=0: ups_le0 += 1
            if v<0: ups_neg += 1
            # coarse bucket
            if v<=0: b='<=0'
            elif v==1: b='1'
            elif v<=5: b='2-5'
            elif v<=20: b='6-20'
            elif v<=100: b='21-100'
            elif v<=1000: b='101-1000'
            else: b='>1000'
            hist[b]+=1
            if n_train % 200 == 0:
                sample_ups.append(v)
        if n % 500000 == 0:
            print(f"...{n} rows", file=sys.stderr)

sample_ups.sort()
def q(p):
    if not sample_ups: return None
    return sample_ups[min(len(sample_ups)-1, int(p*len(sample_ups)))]

print("== TOTAL ==", n)
print("train:", n_train, " test(ups NaN):", n_test, f" test_frac={n_test/n:.3f}")
print("== UPS (train) ==")
print(f" mean={ups_sum/max(1,n_train):.3f} min={ups_min} max={ups_max}")
print(f" ==1: {ups_eq1} ({ups_eq1/max(1,n_train):.3%})  <=0: {ups_le0} ({ups_le0/max(1,n_train):.3%})  <0: {ups_neg}")
print(" quantiles(sample):", {p:q(p) for p in (0.5,0.75,0.9,0.95,0.99)})
print(" hist:", dict(hist))
print("== NETWORK ==")
print(" unique authors:", len(authors), " [deleted]/empty author rows:", deleted_author)
print(" unique link_id (threads):", len(links))
print(" subreddits:", dict(subs.most_common(10)))
print("== BODY ==")
print(f" mean_len={body_len_sum/max(1,n):.1f} max_len={body_len_max} empty_body={empty_body}")
