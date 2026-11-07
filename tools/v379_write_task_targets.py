"""Write task-specific targets and hypotheses from verified feedback; no fitting."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
NAMES={'caco2_wang':'Caco2通透性','ld50_zhu':'LD50','lipophilicity_astrazeneca':'脂溶性',
       'ppbr_az':'PPBR蛋白结合率','solubility_aqsoldb':'溶解度','clearance_hepatocyte_az':'肝细胞清除率',
       'clearance_microsome_az':'肝微粒体清除率','half_life_obach':'半衰期','vdss_lombardo':'VDss'}
HYPOTHESES={
 'caco2_wang':'小样本高方差：小容量与参数平均，保留理化和其他任务观测；后续真实OOF物理残差。',
 'ld50_zhu':'中位数与尾部分位数辅助；不把稀有高误差样本直接删除；检查结构域转移。',
 'lipophilicity_astrazeneca':'保留已有效物理残差支路，研究官方数据内理化先验和小模型互补；外部权重不新增。',
 'ppbr_az':'有界原单位输出、冲突重复测量诊断；分位数辅助不等同去噪或测量条件识别。',
 'solubility_aqsoldb':'保留现有有效局部修正；原单位中位数目标，检验条件异质性与分位数专家。',
 'clearance_hepatocyte_az':'原单位L1与排序联合，解决纯MAE改善但Spearman下降；同份预测双指标验收。',
 'clearance_microsome_az':'联合排序目标与小模型正则化；保留树MAE改善信号但不直接宣称能提分。',
 'half_life_obach':'小样本大波动，参数平均、分位数辅助和排序约束；保留失败验证和所有种子。',
 'vdss_lombardo':'同时评估物理单位误差和排序；不把校准秩基准的MAE下降冒称旧原单位误差下降。'}


def main():
    data=json.loads((ROOT/'docs/v377_task_scope_audit.json').read_text())['rows']
    lines=['# 九项连续任务目标与专家路线','','官方已知最好51.271；目标是9项相对MAE降幅等权平均至少5%，同时单独报告5项直接MAE计分。以下5%逐项数值是目标，不是预测。','',
           '|任务|官方指标|最新官方值|下降5%的MAE目标|官方训练行数|当前专家假设|',
           '|---|---|---:|---:|---:|---|']
    for row in data:
        target=row['five_percent_official_MAE_target']
        lines.append('|'+ '|'.join([NAMES[row['task']],row['official_metric'],f"{row['official_raw']:.9f}",
                     f'{target:.9f}' if target else '需原单位参考；另须排序提高',str(row['counts']['train']),HYPOTHESES[row['task']]])+'|')
    lines += ['', '其余13项为分类：8项AUROC、5项AUPRC。其概率MAE不等价于官方排序指标，不虚构十多项MAE计分任务。', '',
              '4项排序任务的历史输出任意尺度，当前MAE研究用拟合集标签分位数校准的历史秩为参考；不能据此恢复历史模型真实误差。v379对同一原单位融合输出计算MAE、Spearman并写CSV。', '',
              '验收：完整预算后fold0/1唯一选型，冻结再看fold2/valid；失败任务不换配方。采用任务四分区MAE都须下降，4项排序任务还须四分区Spearman均提高。平均降幅的分母始终为9或5，未采用项为零。', '',
              '目前v377/v378无可采用更新；v379尚在训练。公开验证长期复用，局部改善与隐藏集改善不是同一事实。', '']
    (ROOT/'docs/V379_TASK_TARGETS_AND_EXPERT_ROUTES.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(tasks=len(data),goal='Mean relative MAE reduction >=5percent',new_score_predicted=False)))


if __name__=='__main__':main()
