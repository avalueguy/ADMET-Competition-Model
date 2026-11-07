# Edge 提交接入

用户已明确授权：新版本完成验证后，通过当前登录的 Edge 账号提交小分子药物预测比赛 CSV，读取官方成绩，并提供对应文件。授权不包含购买服务器、修改账号或接受新的协议。

已实测：Edge 扩展连接正常；比赛页面处于登录状态；个人中心可显示全部 67 条历史成绩及 22 项指标；赛事“提交方法 → 上传”可打开数据集上传页，存在文件选择控件。此次接入检查没有选择或上传任何文件。

当前可见最高成绩为 **51.166**，提交时间 **2026/10/27 17:22**，22 项指标与本地 v145 成绩记录完全一致。平台未暴露上传文件哈希，v145 对应关系仍依据交付记录与分项匹配。

之后每次提交记录版本、CSV 路径、SHA256、上传时间和平台回执。评分延迟时先标记“待评分”，不能将旧成绩当作本次结果。结果可读时保存完整分项并与基线比较，提供该次文件。

历史最佳分项的回溯均值为 51.190901864，比当前分项均值高 0.024774364。它不是已实测组合，也不表示相关源文件都能恢复；仅说明靠历史挑选不足以达到 52。Caco-2、肝微粒体清除率、hERG、VDSS 是四个存在历史差距的任务。

本轮六路线训练仍在推进，沿用已冻结的选择和确认门槛。没有通过的结果不改名为新提交，不上传同一底稿冒充新模型。

完整原始记录保存在 `platform_history_edge_20261027.json`；结构化对照保存在 `platform_history_edge_audit_20261027.json`。

2026-10-27 后续状态：v172/v173 新模型均未通过冻结确认。v174 在源文件审计中被拒绝：v23 的 Caco-2 和 v46 的微粒体清除率预测已与 v145 相同，v114 的 hERG 官方分项 49.677228 低于 v145 的 49.832263；此前 51.188 估算作废，没有上传。现启动 v175 共享图网络与指纹混合模型，v176 独立验证与归档流水线已接入；详见 CURRENT_STATUS.json 与 docs/v175_followup.json。不得将 rejected 文件作为提交候选。

2026-10-27 新模型跟进：v175 共享图网络与 v178 冻结图表示提升树均未通过确认，v176/v179 未生成 CSV、未上传。两包全部成员及依赖已本地哈希校验。当前 v181 对五个 MAE 任务尝试按叶同现加权中位数预测的分位数森林，沿用 v148 门槛，v182 验证归档流水线已启动；跟进 docs/v181_followup.json。

连接更新：v181 四个任务的互不重叠预计算已完成，原进程恢复后 SSH 和独立 TCP 检查均失败。远端训练状态未知，没有确认新候选或上传。先恢复连接并查原进程，不能盲目重启。

2026-10-28 用户更新：自动提交改为仅在存在真正新版本、固定验证有充分且稳定的提分证据时触发，再获取本次官方成绩单；15分钟检查不触发定时上传。昨晚没有提交新文件。新 SSH 端口25870已连接，确认是空白替换实例，原目录和旧进程不存在；从本地不可变数据、原v146缓存及原v181源恢复未完成实验，失败路线不重跑，门槛不放宽。以 docs/v188_recovery_status.json 和 docs/v181_followup.json 为准。

2026-10-28 新路线v189：LightGBM4.7.0按叶生长提升树，两种固定视图/配置、两个种子，全部21个可改善任务（HIA保护）。21任务105分区及原基线缓存检查通过，标签隔离/独立重载smoke通过；训练PID3565、v191验证归档PID3566。恢复v181仍并行运行。两条路线均使用v148冻结确认门槛，未生成新候选或上传。跟进 docs/v189_followup.json。

2026-10-28 follow-up: v181/v182 completed with no candidate (all five endpoints below frozen development threshold). Archive SHA256 5d707e44cbd83178adec4aa8c396a0cfe337ea8287b209d524d1a6495e367db2; all94 members and dependency hashes verified locally. No CSV/no upload. v189 remains training;14/21 development tasks complete.

2026-10-28 follow-up: v189/v190 completed all21 development endpoints; five frozen nominees, accepted []. CYP3A4 substrate had positive point gains but valid bootstrap90 lower bound -0.042501; other nominees regressed on confirmation. No fallback, CSV or upload. Archive 2b4029155c8616affbd035724d560390edd612d1eb2f0cb3707b57ef2e48c65e,331 members and dependency verified locally. See docs/v190_final_status.json.

2026-10-28 new route v193/v194: randomized isomeric SMILES sequence CNN, four enumeration augmentations and two fixed seeds, fit-only token vocabulary/descriptor scaling, trained from scratch for all21 improvable tasks. Label-isolation/padding/finite-gradient smoke PASS. Training5754 and verification/archive5755 started; v148 gates unchanged, no candidate or upload. See docs/v193_followup.json.

USER PAUSE 2026-10-28T05:55:36.582089+00:00: user requested training stop until their next continue message. Read docs/USER_PAUSE.json before any action; no training, resume, candidate generation or automatic upload while paused.

USER RESUME 2026-10-28T06:22:58.708757+00:00: user explicitly asked to continue training toward52. Prior pause revoked. Verified unchanged server boot_id, process start_ticks, script paths and frozen source hash, resumed original training5754 and pipeline5755 via SIGCONT. No new process or upload. Follow docs/v193_followup.json and docs/v193_resume_state.json; original stable-validation submission policy remains.

2026-10-28 v193/v194 completed: all21 development tasks, two nominees (solubility, bioavailability), accepted[]. Solubility valid relative-MAE gain -0.013347; bioavailability fold2/valid AUROC gains -0.054645/-0.049834. No fallback or CSV/upload. Archive SHA301c364367b3e13e6c0c1a9c441bb9900c10517995afdd6c716d9e5c8ab7a3f4;251 members and dependencies verified locally. See docs/v194_final_status.json.

2026-10-28 new v197/v198: penalized quadratic B-spline additive molecular response model, fit-only feature screening/knots and explicit missingness; logistic/smooth-absolute/rank-normal objectives, two fixed penalties. Label-isolation, monotonic optimizer and closed-form reference smoke PASS. Training8006, pipeline8007, originalv148 gates unchanged. Follow docs/v197_followup.json, no new CSV/upload.

2026-10-28 v197/v198 failure: development stopped at solubility because smooth-absolute IRLS failed the frozen150-iteration convergence assertion. The waiting verifier was terminated, no incomplete result promoted, no CSV/upload. Original source and gates unchanged; preserve failure archive and do not restart this frozen route. Follow docs/v197_followup.json.

2026-10-28 v200/v201 numerical repair: actual failed solubility fold1 fit had IRLS curvature overestimation (median253x), no query scoring used for diagnosis. Exact smooth-absolute Newton curvature preserves objective, original strengths/blends and v148 gates; numerical derivative/BFGS reference and label-isolation smoke PASS, old failing fit converged15 iterations gradient3.76e-10. Only five unfinished MAE endpoints; new training8385, verifier/archive8386. Originalv197 source/failure retained. Read docs/v200_followup.json; no CSV/upload. Automation tool still Transport closed, stale scheduled prompt must defer to current local state.

2026-10-28 v200/v201 complete: exact-Newton repair resolved solver issue, all five MAE tasks completed but none passed unchanged development threshold. No confirmation nominees, CSV or upload. Archive SHA b89817c8b0cf6b7c9736f5a0b387415d6ffe7b539a884dae39c9a23ef47efe75,121 members and dependencies verified locally. Both8385/8386 exited, no active training. Official best remainsv14551.166. Do not rerun additive route; see docs/v201_final_status.json.

2026-10-28 newv204/v205: clamped label diffusion on a symmetric32-neighbor molecular graph, six countTanimoto views plus fit-scaled augmented chemistry/context. Query structures form unlabeled nodes; only actual fit labels supply boundaries/prior, no old predictions. Two fixed strengths .8/.95 and originalv148 blends/gates, all21 improvable tasks with HIA protected. AUROC/MAE/Spearman label-isolation, maximum principle and independent dense-solve smoke PASS. Training8852, verification/archive8853. Followdocs/v204_followup.json; no candidate or upload.

2026-10-28 v204/v205 completed21 tasks: CYP2D6/CYP2C9 substrate nominees both rejected. CYP2D6 four point gains positive but bootstrap90 lower fold2 -0.015816 and valid -0.030047; CYP2C9 regressed. No fallback/CSV/upload. ArchiveSHA23ceac4f6c0f6ff329d4b5628e7da988964bc7aed10a170141266e431ffe8295,325 members and dependencies verified locally. Both8852/8853 exited. Followdocs/v205_final_status.json, retainv14551.166.

2026-10-28 newv207/v208: label-free denoising autoencoder from scratch on34610 unique original official train structures, two fixed6000step seeds,128D each. Only molecular fingerprint/physchem reconstruction, no endpoint labels/old predictions/external weights. Frozen256D embeddings augment21 endpoint CatBoost experts. Label perturbation/finite-gradient/row-order smoke PASS. Structure overlap across endpoint folds disclosed, no endpoint labels used in pretraining. Training9284/verifier9285; originalv148 gates/blends unchanged. Followdocs/v207_followup.json. No CSV/upload.

2026-10-28 v207 follow-up07:58UTC: both label-free6000step seeds completed; full embedding independent reload and all-endpoint-label perturbation PASS.2/21 endpoint development tasks complete, training9284 active/pipeline9285 waiting; process identity/start ticks/boot ID verified. No confirmed CSV/upload. See docs/v207_followup.json.

2026-10-28 v207/v208 complete21 tasks: CYP3A4 substrate and CYP2C9 substrate nominees both rejected. fold2 gains -0.008799 AUROC/-0.006939 AUPRC, bootstrap90 lower bounds negative. No fallback/CSV/upload. ArchiveSHA87f5e5396350eba98804e2a756d873605171a920573376d08563e7ced030a4e6,252 members and dependencies verified locally, both encoders independently reloaded. Both9284/9285 exited; current training count0. Official best remainsv14551.166. Human status question answered: no eligible new submission. See docs/v208_final_status.json.

2026-10-28 newv210/v211: neutral organic-parent and bounded canonical-tautomer countfingerprints/descriptors, plus original chemistry/context. Two fixed CatBoost seeds,1000treesdepth6,all21 improvable endpoints/HIA protected. Salt and keto/enol invariance, originalSMILES mapping, query-label isolation and actual classification/MAE/Spearman independent reload smoke PASS. Actualtraining13106 and verifier/archive13163 verified by bootID/startticks. Gates unchanged, noCSV/upload. Followdocs/v210_followup.json; earlier heartbeatv181/v189 is stale.

2026-10-29 USER POLICY CHANGE: automatic submission revoked. Continue training and full original stable-validation gates; deliver eligible newCSV/path/SHA256/evidence to user for manual submission. Do not upload through Edge or any API. This overrides earlier automatic-upload instructions. NewSSH port24365 uses a different bootID with older retained inputs, no v210 artifacts or active process before recovery. Previousport25870 connection refused; previousv210 final outcome unknown, last verified12/21development only. Original frozenv210 source and signature copied unchanged; fresh smokePASS, training1575 active on replacement instance.

2026-10-29 v210/v211 completed: 21development tasks84models,3nominees allrejected. hERG four positive point gains but both bootstrap90 lower negative; CYP3A4substrate fold2 -0.005510; CYP2C9substrate fold2/valid -0.033352/-0.070873. NoCSV/upload. ArchiveSHA03fe0731348857e5b6e54b7399e3d5de01ded4186f8c06f6bc0e2513b4b797f6,266members and dependency locally verified. Processes1575/1676exited; no rejected-route rerun. See docs/v211_final_status.json.

2026-10-29 newv216/v217: ordered CatBoost with fit-only scaffold-balanced risk,2seeds700treesdepth5,fulloriginalchemistry/shape/pharmacophore/context. SmokePASS,sourcefrozen;training5795,pipeline5796,parallelcoordinator5827,3disjointworkers5836/5837/5838. Originalmainparkedforscheduling,willresumeonlyafterallworkers0. Manualdeliveryonly. Followdocs/v216_followup.json.

2026-10-29 v216/v217 completed:21developmenttasks84models,2nomineesbothrejected.CYP3A4substrate allfourpointgainspositive butbootstrap90lowerfold2 -0.004698/valid -0.009147;CYP2C9substratefold2 -0.007611/valid -0.012546.NoCSV/upload.ArchiveSHA636778ab3083c45f9055f24ad469ea74c176dd6b84b4f939a88a623b2cb65d0c,266membersanddependency locallyverified.Alltrainingprocessesexited.Docs/v217_final_status.json.

2026-10-29 USER QUIET POLICY and newv221/v222: realtimeprogress cancelled; eligibleCSV manualdelivery only,no upload/scorefetch. Fragmentfactorization machine learns explicit rank32 pairwise fragment/context interactions from scratch,two fixedseeds100epochs. Analyticalformula, querylabelisolation, finitegradients, exactCPU checkpointreplay smokePASS. Sourcefrozen, originalv148gates/105cachesverified. Training12026/pipeline12027 active; noeligibleCSV yet. Followdocs/v221_followup.json.

2026-10-29 v221/v222 completed21developmenttasks84models and3frozenconfirmations,allrejected. PGP,CYP3A4substrate,bioavailability had negativefold2/validgains. NoCSV/upload/fallback.307members anddependencySHA locallyverified;archiveSHA4a8cb879beb5baa3b03eeb2406d6fc1327b48a6333e93674e068037ad0dbbce2. Both12026/12027exited;do notrerun. Docs/v222_final_status.json.

2026-10-29 newv224/v225: grouped fullrank PCA rotation forests learn oblique chemical splitdirections,2fixedseeds/8rotations/24trees perrotation;fit-only allpreprocessing/rotations,no oldmodels/predictions. Querylabelisolation,orthogonality/distance preservation,threeactualmetrictypesmodelreload,roworder smokePASS. Frozenv148gates unchanged. Driver12754/3disjointdevworkers12770/12771/12772,pipeline12755active. Manualdeliveryonly;quietuntilqualifiedCSV. Followdocs/v224_followup.json.

2026-10-29 v224/v225 completed:21development tasks84seedbundles,4frozen nominees allrejected. Hepatocyte andCYP3A4substrate all4pointgains positive butbootstraplower support fails.267members/dependencyhash verified,archiveSHAbda7582e432629a62d0a26ba90a8f355972aa4080989a9f4ef47ea3e080ca255. Allprocesses exited;noCSV/upload/fallback. See docs/v225_final_status.json. Userrequests targetedoptimization anchored onv145.

2026-10-29 userdirected v145incrementaloptimization: newv228/v229 honestinner-scaffold-crossfit correction for6lowest officialendpoints;other16protected. NewreferenceOOF is explicitlysurrogate,notexactv145OOF;nohistoricalproxy/in-samplefitoffset.450depth5references and250depth3boundedcorrections,twofixedseeds,unchangedv148gates. Threeclassification/MAE/Spearman actualreload/querylabelisolation/OOfdisjoint/correctionbound smokePASS. Preflight105cacheanddependencySHA/frozensourcesverified;driver19850,pipeline19851,workers19859/19860/19861active. NoCSV/upload. Followdocs/v228_followup.json.

2026-10-29 v228/v229 completed6targetedv145incrementalcorrectiondevelopment tasks,36honestinnerOOFreferences/24correctionmodels.Nonepassedoriginaldevelopmentpracticalthreshold;no confirmation/CSV/upload.142membersanddependencySHA verifiedlocally;archiveSHAbcae36c83508818d9fd86608c3d698e4f8ec2d3f17116d8dc0b992fba5ad2932.Allprocessesexit0.No retraining/fallback. See docs/v229_final_status.json.

2026-10-29 newv233/v234 targetedv145incrementalexperts: cross-scaffoldsupervisednonnegative15channelkernelalignment plusSVC/SVR. Differsfromfixedv139/v150kernelsandoldv103differencemodel. Sixlowestofficialv145tasks,other16protected;originalv148runner/gatesunchanged. NNLSclosedform/KKT,PSD,querylabelisolation,actualthree-metrictypecheckpointreload smokePASS.105cache/dependency/frozenfilesverified. Driver21544/pipeline21545started;followdocs/v233_followup.json. NoCSV/upload.

2026-10-29 v233/v234 complete6targetdevelopmenttasks12models,onehERGnomineeall4positive AUROCgains(.003105,.019135,.014198,.004464)butvalidCI90lower-.003877,notaccepted.100members/dependencyverified,archiveSHAbd899cbcd17e391aa91fe35eac0e8db0b7492ab13b94e5bd7e7c1f983140a602.NoCSV/upload/fallback. Docs/v234_final_status.json.

2026-10-29 newv238/v239 targetedv145viewattentionexperts,6lowestofficialtasks/other16protected. Sixfingerprinttokens+physical/contexttokens self-attend2layers4heads64D;2fixedseeds100epochs900..3000steps.Nooldmodels/predictionsorpretrainedweights. ActualCUDAgradientstep,3metrictypesquerylabelisolation/exactcheckpointCPUfloat64reload/roworder smokePASS.105cache/dependency/frozenfilesverified;driver21878/pipeline21879started,singleGPUworker.NoCSV/upload;originalv148gatesunchanged. Followdocs/v238_followup.json.

2026-10-29 v238/v239 completed6targetdevelopmenttasks12partitions24seedmodels. Everytargethadnonpositivegainonatleastonedevelopmentfold;nonominee/confirmation/CSV/upload. Allprocessesexit0.104members/dependenciesverified,archiveSHA6b5d3db1b6b7e72de6aea5f5a86e65f58e26983b26dd295e31d63f0b17907d36. Nextmodelnotstarted;retainofficialbestv14551.166. See docs/v239_final_status.json.

2026-10-29 HK17:04: v238/v239 complete rejected; v243 read-only v145 error/validation feasibility review complete. New v244/v245 conditional logistic mixture density for Caco2/PPBR/lipophilicity only (19 protected) now training; 3-task label-isolation/CDF reference/CUDA update/save-reload/row-order smoke PASS,105 baseline caches and frozen sources/dependencies verified. Driver23649 pipeline23650 worker23657; identities in docs/v244_remote_start.json. Original v148 source/gates unchanged, no fallback. Manual CSV delivery only; no automatic upload or score retrieval. Follow docs/v244_followup.json.

2026-10-29 HK17:10: v244/v245 completed 3 tasks/6 development partitions/12 models; all3 had at least one nonpositive development gain, no confirmation nominee or CSV. Driver23649/pipeline23650/worker23657/selector23823 exited0. Local release83members+dependencies PASS, SHA02873f10f067d9fe052670b3baf9d66830ac8166ada1b245a172b8b180498b0b. v249 development-only chemical-strata audit completed: no supported stable correction for Caco2/lipo; PPBR mean vs median biases conflict. See docs/v249_review_decision.json. No next model started. Manual delivery/no auto upload unchanged.
