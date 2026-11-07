import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

const root = 'C:/Users/lenovo/Documents/ChatGPT/多媒体/tdc_overlap_diagnostic';
const report = JSON.parse(await fs.readFile(`${root}/overlap_report.json`, 'utf8'));
const records = JSON.parse(await fs.readFile(`${root}/matched_rows.json`, 'utf8'));
const template = await fs.readFile(report.template);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
if (hash(template) !== report.template_sha256) throw Error('Template hash changed');
const wb = await Workbook.fromCSV(template.toString('utf8').replace(/^\uFEFF/, ''), {sheetName:'Public label diagnostic'});
const sheet = wb.worksheets.getItemAt(0);
sheet.getRange(`E2:E${records.length+1}`).values = records.map(r => [Number(r.public_label)]);
wb.recalculate();
const predictions = sheet.getRange(`E2:E${records.length+1}`).values;
for (let i=0; i<records.length; i++) {
  if (predictions[i][0] !== Number(records[i].public_label)) throw Error(`Prediction mismatch ${i}`);
}
const spans = raw => {
  let start=raw.subarray(0,3).equals(Buffer.from([239,187,191]))?3:0;
  let quoted=false, record=[], result=[];
  for(let i=start; i<raw.length; i++) {
    if(raw[i]===34) {
      if(quoted && raw[i+1]===34){i++; continue;}
      quoted=!quoted;
    } else if(!quoted && [44,10,13].includes(raw[i])) {
      record.push([start,i]);
      if(raw[i]!==44){result.push(record); record=[]; if(raw[i]===13 && raw[i+1]===10)i++;}
      start=i+1;
    }
  }
  if(quoted)throw Error('Unclosed quote');
  if(start<raw.length || record.length){record.push([start,raw.length]);result.push(record);}
  return result;
};
const fields=spans(template);
if(fields.length!==records.length+1 || fields.some(r=>r.length!==6))throw Error('Template shape');
const chunks=[];let cursor=0;
for(let i=1;i<fields.length;i++) {
  const [lo,hi]=fields[i][4];
  if(!['','""'].includes(template.subarray(lo,hi).toString()))throw Error('Nonempty template prediction');
  const value=String(predictions[i-1][0]);
  if(!Number.isFinite(Number(value)))throw Error('Nonfinite value');
  chunks.push(template.subarray(cursor,lo),Buffer.from(value));cursor=hi;
}
chunks.push(template.subarray(cursor));
const payload=Buffer.concat(chunks), actual=spans(payload);
const reverse=[];cursor=0;
for(let i=1;i<actual.length;i++) {
  const [lo,hi]=actual[i][4], [oldlo,oldhi]=fields[i][4];
  reverse.push(payload.subarray(cursor,lo),template.subarray(oldlo,oldhi));cursor=hi;
}
reverse.push(payload.subarray(cursor));
if(!Buffer.concat(reverse).equals(template))throw Error('Nonprediction bytes changed');
const output=`${root}/outputs/我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv`;
await fs.mkdir(`${root}/outputs`,{recursive:true});
await fs.writeFile(output,payload);
const audit={status:'PASS',rows:records.length,tasks:report.task_count,source_type:report.data_type,
  source_zip_sha256:report.source_zip_sha256,template_sha256:report.template_sha256,
  output,output_sha256:hash(payload),bytes:payload.length,nonprediction_bytes_exact:true,
  all_predictions_equal_public_observations:true,unmatched_rows:0,automatic_platform_submission:false};
await fs.writeFile(`${root}/diagnostic_file_verification.json`,JSON.stringify(audit,null,2)+'\n');
console.log(JSON.stringify(audit,null,2));
console.log((await wb.inspect({kind:'table',range:'Public label diagnostic!E1:F5',include:'values',tableMaxRows:5,tableMaxCols:2,maxChars:2000})).ndjson);
// Visual check of a compact imported range; machine CSV bytes above preserve the template.
sheet.getRange('E1:F5').format.columnWidth=18;
sheet.getRange('E1:F5').format.font={name:'Arial',size:10};
sheet.getRange('E2:E5').format.numberFormat='0.000000';
const preview=await wb.render({sheetName:'Public label diagnostic',range:'E1:F5',scale:1.5,format:'png'});
await fs.writeFile(`${root}/diagnostic_preview.png`,new Uint8Array(await preview.arrayBuffer()));
