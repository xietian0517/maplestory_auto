

## 2026-09-25 round067–068：下跳往返与假超时

067：180s hybrid，最新ActiveRecovery尊重anchor/jump等待；净经验3101/179.5437857s=1036.293/min，967提交边沿全部审计匹配、外部0、HUD未知0。总231.49s，parked，native图确认crown松键。靠近起跳44.53s，attack27.39s。等待结束分别由可见攻击、边缘、人物丢失或预算到期触发，未发现通用恢复在等待内插入走路。

068：补齐经原图核对的底层左端72px；world_geometry observed_14.left=-380（原-308），证据066 frame5023/camera=[380,-430]，图00_middle及其它连续图显示木质地面延至screen0。geometry_review.json验证同一人物脚点原floor=None、修正后observed_14。没有扩张其它未核对平台。

下跳修复：067 frame2244/2250/2268记录x405.5→373→428，原程序到中点才刹车，明明已位于347..453落地范围却反向追中点。control.py增加提前释放方向与平台相对位置连续.25s漂移≤5px稳定判据；稳住后只有源平台15px/目标24px内边距、离绳口≥24px才允许原地Down→Alt，否则重新靠近。ActiveRecovery尊重所属drop_brake等待。

068结果：3788/179.5379976s=1265.916/min；970边沿全部匹配、外部0、HUD未知0。总202.77s，安全crown完成。靠近起跳16.88s，attack38.30s，drop_brake4.83s。单段变化不是因果证明，也未达人工2330.117/min。

068运行期间修复下一轮路线计时：_navigate_step不再为空中transition写pending_since，新选path从当前时间起算；两个回归测试先失败再通过。069开始600s完整实测，加载这条修复。

人工攻击可见性诊断：human_attack_visibility_diagnostic.json从10Hz状态/按键核算攻击246.881s，其中附近local目标129.074s、仅远端检测21.37s、附近未检测62.732s、人物未知33.705s。粗时长因10Hz采样与原245.318s边沿统计不同；不能视为伤害/击杀。人工76.60s、102.22s原图包含遮住精灵的命中光效；51.10s右侧边缘目标不全，保留未知。

069运行期间准备的下一轮改动（069未加载）：hybrid站立攻击后0.18s内短漏检先attack_reacquire_wait，不按键、不启动新攻击、不更新等待期限。需真实已提交攻击、人物有效、位置变化≤12x/8y、朝向未变；重新看见目标正常攻击，超时正常导航。068 frame166→167（40ms）怪物仍清晰存在/有残余HP，识别却消失并切approach_launch；相关原图在068/failure_review/attack_gap_*.png。没有放宽识别阈值或把消失认定死亡。


## round069完整长测与0.4.0交付

净EXP+8415/599.6803225s=841.949/min，2881输入提交全匹配，计分区间外部0，HUD2788/未知0。690.49s返回顶部crown确认阶段超时，原报告parking_unconfirmed保留；native随后确认crown松键。长测仍未达到人工2330.117/min。已交付dist/MapleAIController-v0.4.0.exe及完整试用包，EXE本机无游戏按键自检通过、预设ready/FFmpeg有效。源码版本保存在releases/v0.4.0/source.zip。用户最终要求交付后暂停，保留目标，已开始自己试用GUI，不干预其运行。
