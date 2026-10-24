# T2_6 undo: drop the 21 docs restored from e11ece8^ and 6355e93^.
# They were additions, so the reversal is a removal from git, not a file move.
$ErrorActionPreference='Stop'
Set-Location 'E:\Workspaces\RetNet'
git rm --quiet -- docs/02_训练管线.md docs/04_路线与扩展研究.md docs/06_清理与运维.md `
  docs/08_chunkwise_growth_research.md docs/09_272M_v10_方案备份.md `
  docs/11_R1思维链与长上下文.md docs/12_训练状态盘点.md `
  docs/13_基准测试重设计与评测结果.md docs/17_CoT训练方法.md docs/18_架构替代方案.md `
  docs/21_训练数据与算力盘子.md docs/23_0.5B扩容方案.md docs/24_AutoDL上机手册.md `
  docs/26_RetNet-0.5B_技术报告.md docs/29_v20训练计划评审.md `
  docs/30_v21训练监控与判据.md docs/31_v21训练报告.md docs/32_当前状态与接手须知.md `
  docs/33_失败合集.md docs/34_交接_给下一个Agent.md docs/INDEX.md
"undo done: 21 restored docs removed from the index. Commit to record it."
