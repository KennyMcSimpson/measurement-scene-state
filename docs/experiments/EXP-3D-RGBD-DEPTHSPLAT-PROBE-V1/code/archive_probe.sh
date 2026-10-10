#!/bin/bash
# Archive the exploratory DepthSplat probe (DEV + EVAL34) into the repository: code, small result files, logs,
# provenance hashes. Predictions (npz), checkpoints and the DepthSplat checkout stay outside git.
set -eu
d=~/cv_work/depthsplat
out=~/zonghan_o_mirror/measurement-scene-state/outputs/EXP-3D-RGBD-DEPTHSPLAT-PROBE-V1
mkdir -p "$out/code" "$out/results/probe_dev" "$out/results/probe_eval34" "$out/logs"
cp -p $d/{ds_predict,ds_score,ds_diag,ds_selfcheck,ds_finetune,ds_summary,ds_fuse,make_probe_root,make_probe_root_eval}.py "$out/code/"
cp -p $d/{gpu_probe,selfcheck_job,extra_job,ft_job,eval_ft_job,chain_v2,chain_align,align_job,score_all,e34_job,e34g_job,e34_score,rescore,archive_probe}.sh "$out/code/"
cp -p ~/cv_tools/cv_gpu_run.sh "$out/code/cv_gpu_run.sh.copy"
for p in probe_dev probe_eval34; do
  cp -p $d/$p/evaluation_contract.json $d/$p/manifest_*.json "$out/results/$p/"
  cohort=$(ls $d/$p/raw/eval)
  for f in prediction_seal GT_access_before_seal state_hashes; do
    [ -e $d/$p/raw/eval/$cohort/$f.json ] && cp -p $d/$p/raw/eval/$cohort/$f.json "$out/results/$p/"
  done
done
cp -p $d/ds_summary_dev.json $d/ds_summary_e34.json $d/ds_summary_e34g.json "$out/results/"
for v in $d/ds_dev_*/ $d/ds_e34_*/; do
  name=$(basename "$v")
  mkdir -p "$out/results/$name"
  for f in run_info.json scores.json diag.json score.out fuse_tol0.05_k2.json; do
    [ -e "$v$f" ] && cp -p "$v$f" "$out/results/$name/"
  done
done
for r in ft_v1 ft_v2; do mkdir -p "$out/results/$r"; cp -p $d/$r/finetune_log.json "$out/results/$r/"; done
for f in selfcheck_none selfcheck_w2c selfcheck_pixel_center ft_v1 ft_v2 chain_v2 chain_align e34_job e34g_job e34_score probe_eval34_seal; do
  [ -e $d/$f.log ] && grep -v -E "FutureWarning|impl_abstract" $d/$f.log > "$out/logs/$f.log" || true
done
cp -p $d/_attempts/e34_job.attempt1.log "$out/logs/e34_job.attempt1_det_assert.log"
grep -v -E "FutureWarning|impl_abstract" $d/_attempts/ds_e34_rgbd_x9.attempt1_det_assert.log | tail -20 > "$out/logs/ds_e34_rgbd_x9.attempt1_det_assert.tail.log"
{
  echo "DepthSplat checkout: $d/depthsplat @ $(git -C $d/depthsplat rev-parse HEAD)"
  echo "DepthSplat weights: $(sha256sum /home/xmum26_zonghan/zonghan_o_mirror/_models/depthsplat/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth)"
  echo "Rasterizer source: $(tr '\n' ' ' < $d/diff-gaussian-rasterization-modified/SOURCE_COMMIT)"
  (cd $d && sha256sum dgr_*.tar.gz glm_*.tar.gz)
  echo "DINOv2 hub code: ~/.cache/torch/hub/facebookresearch_dinov2_main (code tree sha256 117c2d9278ba282500e5d3924dae634672bd19cf892f3f8f4800fbd80b7b6af4, equal to V31's)"
  echo "Fine-tuned checkpoints (outside git):"
  sha256sum $d/ft_v1/depthsplat_ft_final.pth $d/ft_v2/depthsplat_ft_final.pth
  echo "Environment: ~/envs/depthsplat, $(~/envs/depthsplat/bin/python -c 'import torch; print("torch", torch.__version__)')"
  echo "Predictions (npz, outside git): $d/ds_dev_*/ and $d/ds_e34_*/; failed attempt kept in $d/_attempts/"
} > "$out/provenance.txt"
(cd "$out" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
du -sh "$out"; find "$out" -type f | wc -l
