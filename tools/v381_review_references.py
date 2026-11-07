"""Read selected primary implementation references; no model fitting."""
from pathlib import Path
import hashlib,json,re,urllib.request
from html import unescape
ROOT=Path(__file__).resolve().parents[1]
URLS=(
 ('grouped_stacking','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html',
  'Stacking fits final estimator from cross-validated base predictions. Our implementation uses explicit scaffold groups and native MAE/joint LP, not default KFold or default ridge.'),
 ('lp_certificates','https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html',
  'HiGHS exposes primal residuals and marginal dual sensitivities. Retain and independently verify stationarity/complementarity without another fitting run.'),
 ('quantile_map','https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html',
  'Quantile transformations can reduce outlier effects and distort linear correlations. Our explicit fit-label inverse empirical CDF is nondecreasing, may introduce ties, and offers no Spearman/MAE guarantee.'))


def main():
    records=[]
    for key,url,application in URLS:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=25) as response:payload=response.read()
            raw=payload.decode('utf-8',errors='replace')
            title=unescape(re.search(r'<title>(.*?)</title>',raw,re.S).group(1))
            clean=re.sub(r'<script\b.*?</script>|<style\b.*?</style>',' ',raw,flags=re.S)
            clean=unescape(re.sub('<[^>]+>',' ',clean));clean=re.sub(r'\s+',' ',clean)
            if key=='grouped_stacking':phrase='The final_estimator_ is trained'
            elif key=='lp_certificates':phrase='marginals'
            else:phrase='non-linear'
            start=clean.lower().find(phrase.lower());excerpt=clean[max(0,start-100):max(0,start-100)+1500]
            records.append(dict(key=key,url=url,status='READ',title=title,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),excerpt=excerpt,application=application))
        except Exception as exc:records.append(dict(key=key,url=url,status='UNAVAILABLE',error=str(exc)))
    result=dict(scope='Three selected primary library pages, supplements earlier17; not exhaustive literature search.',https_certificate_verification=True,records=records)
    (ROOT/'docs/v381_primary_references.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
