import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

// This is a label-assisted scoring fixture, never model output.
const ROOT = 'C:/Users/lenovo/Documents/ChatGPT/多媒体/tdc_overlap_diagnostic';
const REPO = 'C:/Users/lenovo/Documents/ChatGPT/New project/ADMET-Competition-Model';
const OUT = `${ROOT}/synthetic_offline`;
const SEED = 20261106;
const FRACTION = 0.30;
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const recordsBytes = await fs.readFile(`${ROOT}/matched_rows.json`);
const records = JSON.parse(recordsBytes.toString('utf8'));
const overlap = JSON.parse(await fs.readFile(`${ROOT}/overlap_report.json`, 'utf8'));
const referenceBytes = await fs.readFile(`${REPO}/docs/platform_scorecard_v387_20261106.json`);
const reference = JSON.parse(referenceBytes.toString('utf8'));
const template = await fs.readFile(overlap.template);
if (hash(template) !== overlap.template_sha256) throw Error('Template hash changed');

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
const random = mulberry32(SEED);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
function quantile(values, p) {
  const sorted = [...values].sort((a,b) => a-b);
  const index = (sorted.length - 1) * p;
  const lo = Math.floor(index), hi = Math.ceil(index);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index-lo);
}
function ranks(values) {
  const order = values.map((value,index) => ({value,index})).sort((a,b) => a.value-b.value);
  const result = Array(values.length);
  for (let i=0; i<order.length;) {
    let end = i+1;
    while (end<order.length && order[end].value===order[i].value) end++;
    const rank = (i+end-1)/2 + 1;
    for (let j=i; j<end; j++) result[order[j].index] = rank;
    i = end;
  }
  return result;
}
function pearson(a,b) {
  const ma = mean(a), mb = mean(b);
  let covariance=0, va=0, vb=0;
  for (let i=0; i<a.length; i++) {
    const da=a[i]-ma, db=b[i]-mb;
    covariance += da*db; va += da*da; vb += db*db;
  }
  return covariance/Math.sqrt(va*vb);
}
function metricValue(metric, truth, predicted) {
  if (metric==='MAE') return mean(truth.map((value,i) => Math.abs(value-predicted[i])));
  if (metric==='Spearman') return pearson(ranks(truth),ranks(predicted));
  if (!truth.every(value => value===0 || value===1)) throw Error('Nonbinary classification labels');
  const positives = truth.reduce((sum,value) => sum+value,0);
  if (positives===0 || positives===truth.length) throw Error('Single class');
  if (metric==='AUROC') {
    const ranked = ranks(predicted);
    const positiveSum = truth.reduce((sum,value,i) => sum+value*ranked[i],0);
    return (positiveSum-positives*(positives+1)/2)/(positives*(truth.length-positives));
  }
  if (metric==='AUPRC') {
    // Group equal scores before accumulating sklearn-style average precision.
    const order = predicted.map((value,index) => ({value,index})).sort((a,b) => b.value-a.value);
    let tp=0, seen=0, ap=0;
    for (let i=0; i<order.length;) {
      let end=i+1;
      while (end<order.length && order[end].value===order[i].value) end++;
      const before=tp;
      for (let j=i; j<end; j++) tp+=truth[order[j].index];
      seen+=end-i;
      ap += ((tp-before)/positives)*(tp/seen);
      i=end;
    }
    return ap;
  }
  throw Error(`Unknown metric: ${metric}`);
}
function normalize(metric, raw, baseline) {
  if (metric==='MAE') return 100*baseline/(raw+baseline);
  if (metric==='Spearman') return 100*(1+raw)/(2+raw+baseline);
  return 100*raw/(raw+baseline);
}

const allPredictions = records.map(row => Number(row.public_label));
const selectedMask = records.map(() => false);
const tasks = [];
for (const specification of reference.tasks) {
  const indices = records.flatMap((row,index) => row.task===specification.task ? [index] : []);
  if (!indices.length) throw Error(`No rows for ${specification.task}`);
  if (indices.some(index => records[index].metric!==specification.metric)) throw Error('Metric mismatch');
  const truth = indices.map(index => Number(records[index].public_label));
  const order = [...indices];
  for (let i=order.length-1; i>0; i--) {
    const j = Math.floor(random()*(i+1));
    [order[i],order[j]] = [order[j],order[i]];
  }
  const selected = order.slice(0, Math.round(FRACTION*order.length));
  const classification = ['AUROC','AUPRC'].includes(specification.metric);
  const scale = specification.metric==='MAE' ? specification.reference_score
    : quantile(truth,0.75)-quantile(truth,0.25);
  if (!classification && !(scale>0)) throw Error('Nonpositive perturbation scale');
  for (const index of selected) {
    selectedMask[index] = true;
    const label = Number(records[index].public_label);
    let value = classification ? (1e-12 + (1-2e-12)*random())
      : label + (2*random()-1)*scale;
    if (specification.task==='ppbr_az') value = Math.max(0,Math.min(100,value));
    if (specification.metric==='Spearman' || specification.task==='ld50_zhu') value = Math.max(0,value);
    if (!Number.isFinite(value)) throw Error('Nonfinite synthetic value');
    allPredictions[index] = value;
  }
  const predicted = indices.map(index => allPredictions[index]);
  const raw = metricValue(specification.metric,truth,predicted);
  const perfectRaw = metricValue(specification.metric,truth,truth);
  tasks.push({
    task:specification.task, metric:specification.metric, rows:indices.length,
    reference_score:specification.reference_score, selected_rows:selected.length,
    changed_rows:indices.filter(index => allPredictions[index]!==Number(records[index].public_label)).length,
    method:classification ? 'Selected values replaced by independent Uniform(0,1)'
      : 'Selected labels plus Uniform(-scale,scale); PPBR clipped to [0,100]; LD50 and Spearman tasks clipped at 0',
    perturbation_scale:classification ? null : scale,
    synthetic_raw_metric:raw, synthetic_normalized_score:normalize(specification.metric,raw,specification.reference_score),
    unperturbed_raw_metric:perfectRaw, unperturbed_normalized_score:normalize(specification.metric,perfectRaw,specification.reference_score)
  });
}
const syntheticScore = mean(tasks.map(task => task.synthetic_normalized_score));
const unperturbedScore = mean(tasks.map(task => task.unperturbed_normalized_score));
if (!(syntheticScore>=60 && syntheticScore<unperturbedScore)) {
  throw Error(`Fixture does not meet requested range: ${syntheticScore}, upper bound ${unperturbedScore}`);
}

// Author numeric predictions in Artifact Tool. Serialize only prediction spans
// so identifiers, escaping, row ordering, BOM, and CRLF stay byte-exact.
const wb = await Workbook.fromCSV(template.toString('utf8').replace(/^\uFEFF/,''), {sheetName:'OFFLINE SYNTHETIC'});
const sheet = wb.worksheets.getItemAt(0);
sheet.getRange(`E2:E${records.length+1}`).values = allPredictions.map(value => [value]);
wb.recalculate();
const written = sheet.getRange(`E2:E${records.length+1}`).values;
if (written.some((row,index) => row[0]!==allPredictions[index])) throw Error('Artifact Tool value mismatch');
function spans(raw) {
  let start = raw.subarray(0,3).equals(Buffer.from([239,187,191])) ? 3 : 0;
  let quoted=false, row=[], rows=[];
  for (let i=start; i<raw.length; i++) {
    if (raw[i]===34) {
      if (quoted && raw[i+1]===34) { i++; continue; }
      quoted=!quoted;
    } else if (!quoted && [44,10,13].includes(raw[i])) {
      row.push([start,i]);
      if (raw[i]!==44) { rows.push(row); row=[]; if (raw[i]===13 && raw[i+1]===10) i++; }
      start=i+1;
    }
  }
  if (quoted) throw Error('Unclosed CSV quote');
  if (start<raw.length || row.length) { row.push([start,raw.length]); rows.push(row); }
  return rows;
}
const fields=spans(template);
if (fields.length!==records.length+1 || fields.some(row => row.length!==6)) throw Error('CSV shape');
const chunks=[];
let cursor=0;
for (let i=1; i<fields.length; i++) {
  const [lo,hi]=fields[i][4];
  if (!['','""'].includes(template.subarray(lo,hi).toString())) throw Error('Nonempty template prediction');
  chunks.push(template.subarray(cursor,lo),Buffer.from(String(written[i-1][0])));
  cursor=hi;
}
chunks.push(template.subarray(cursor));
const payload=Buffer.concat(chunks), actual=spans(payload), reversed=[];
cursor=0;
for (let i=1; i<actual.length; i++) {
  const [lo,hi]=actual[i][4], [oldLo,oldHi]=fields[i][4];
  reversed.push(payload.subarray(cursor,lo),template.subarray(oldLo,oldHi));
  cursor=hi;
}
reversed.push(payload.subarray(cursor));
if (!Buffer.concat(reversed).equals(template)) throw Error('Nonprediction CSV bytes changed');

await fs.mkdir(OUT,{recursive:true});
const output=`${OUT}/OFFLINE_SYNTHETIC_seed${SEED}.csv`;
const audit={
  data_type:'PUBLIC_TEST_LABELS_WITH_RANDOM_PERTURBATIONS_NOT_MODEL_PREDICTIONS',
  intended_use:'OFFLINE_SCORING_CHECK_ONLY', competition_compliant:false,
  score_interpretation:'Calculated against the same public labels used to construct this fixture. Not a measure of predictive performance or held-out accuracy.',
  automatic_platform_submission:false, seed:SEED, random_generator:'Mulberry32',
  selection_fraction:FRACTION, row_selection:'Fisher-Yates shuffle independently within each task, round(0.30*n) rows selected',
  selected_rows:selectedMask.filter(Boolean).length,
  changed_rows:records.filter((row,index) => allPredictions[index]!==Number(row.public_label)).length,
  unchanged_rows:records.filter((row,index) => allPredictions[index]===Number(row.public_label)).length,
  rows:records.length, task_count:tasks.length, synthetic_local_score:syntheticScore,
  all_label_upper_bound:unperturbedScore, platform_score:null, platform_acceptance:'NOT_TESTED',
  scoring_formulas:{
    AUROC_AUPRC:'100*raw/(raw+reference_score)', MAE:'100*reference_score/(raw+reference_score)',
    Spearman:'100*(1+raw)/(2+raw+reference_score)', total:'Arithmetic mean over 22 task scores',
    AUPRC_convention:'Average precision, ties grouped; not trapezoidal PR area',
    Spearman_convention:'Pearson correlation of average-tie ranks'
  },
  source_url:overlap.source_url, source_zip_sha256:overlap.source_zip_sha256,
  matched_rows_json_sha256:hash(recordsBytes), template:overlap.template, template_sha256:hash(template),
  reference_scores_file:`${REPO}/docs/platform_scorecard_v387_20261106.json`, reference_scores_file_sha256:hash(referenceBytes),
  output, output_sha256:hash(payload), bytes:payload.length,
  nonprediction_bytes_exact:true, tasks
};
await fs.writeFile(output,payload);
await fs.writeFile(`${OUT}/synthetic_score_report.json`,JSON.stringify(audit,null,2)+'\n');
const readme=[
  '# 离线合成评分测试', '',
  '此文件使用公开 TDC 测试标签加随机扰动生成，不是模型预测，不符合赛事对模型生成预测的要求。仅用于本地验证 CSV 解析、指标和归一化评分计算。', '',
  `本地合成总分：${syntheticScore.toFixed(9)}；未扰动标签理论分：${unperturbedScore.toFixed(9)}。`,
  `固定种子：${SEED}；每个任务随机选约 30% 行，共选择 ${audit.selected_rows} 行，实际改变 ${audit.changed_rows} 行。`,
  '分类选中值替换为独立的 0–1 均匀随机数。MAE 回归任务的扰动为 ±reference_score 均匀分布；Spearman 任务的扰动尺度为标签四分位距。PPBR 截断至 0–100，LD50 和 Spearman 任务截断至非负。其他行保留公开标签。',
  '保留了模板所有非 y_pred 字节，包括分子信息、任务、样本索引、指标、BOM 和 CRLF。结构检查通过不等于赛事规则合规或平台接收。',
  '这类标签辅助合成分数不能测试模型准确性、泛化能力或真实提分效果。没有训练，没有上传，没有修改历史模型或历史提交文件。', '',
  `公开数据来源：${overlap.source_url}`,
  `CSV SHA256：${audit.output_sha256}`, '',
  '完整方法、每项指标、来源哈希及得分见 synthetic_score_report.json。'
].join('\n');
await fs.writeFile(`${OUT}/README.md`,readme+'\n');
console.log(JSON.stringify({output,rows:audit.rows,tasks:audit.task_count,selected_rows:audit.selected_rows,
  changed_rows:audit.changed_rows,synthetic_local_score:syntheticScore,all_label_upper_bound:unperturbedScore,
  sha256:audit.output_sha256},null,2));
console.log((await wb.inspect({kind:'table',range:'OFFLINE SYNTHETIC!E1:F8',include:'values',tableMaxRows:8,tableMaxCols:2,maxChars:1800})).ndjson);
sheet.getRange('E1:F8').format.columnWidth=24;
sheet.getRange('E1:F8').format.font={name:'Arial',size:10};
sheet.getRange('E2:E8').format.numberFormat='0.000000';
const preview=await wb.render({sheetName:'OFFLINE SYNTHETIC',range:'E1:F8',scale:1.5,format:'png'});
await fs.writeFile(`${OUT}/preview.png`,new Uint8Array(await preview.arrayBuffer()));
