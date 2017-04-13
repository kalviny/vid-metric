%flow_root = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/img/fw/';
%score_root = '/data01/kalviny/dataset/MOT/2015/detection/mot_model0_2015_proposal_300/mot_acf_2015_without_nms/'
%output_root = '/data01/kalviny/dataset/MOT/2015/detection/proposal_flow/mot_model0_2015_proposal_300/mot_acf_2015_without_nms';

flow_root = '/data01/kalviny/dataset/kitti/opt_flow/img/fw/';
score_root = '/data01/kalviny/dataset/kitti/detection/kitti_proposal_300/car/kitti_acf_without_nms/';
output_root = '/data01/kalviny/dataset/kitti/detection/propsoal_flow/kitti_proposal_300/car/kitti_acf_without_nms/';

mcs_mgp(flow_root, score_root, output_root)
