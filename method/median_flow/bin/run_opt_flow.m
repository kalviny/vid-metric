
rt_flow = '/data01/kalviny/dataset/kitti/opt_flow/img/bw/';
rt_score = '/data01/kalviny/dataset/kitti/detection/kitti_proposal_300/car/kitti_acf_weighted_nms_pf/';
rt_output = '/data01/kalviny/dataset/kitti/opt_flow/kitti_proposal_300/car/kitti_acf_weighted_nms_pf/bw/'; 

%rt_flow = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/img/bw/';
%rt_score = '/data01/kalviny/dataset/MOT/2015/detection/mot_acf_2015_proposal_300/mot_acf_2015_weighted_weighted_nms_pf/';
%rt_output = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_acf_2015_proposal_300/mot_acf_2015_weighted_weighted_nms_pf/bw/'; 


if ~exist(rt_output, 'dir')
        mkdir(rt_output);
end


%vid_name = {'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17'};
vid_name = {'0011', '0012', '0013', '0014', '0015', '0016', '0018', '0009', '0020'} % car
%vid_name = {'0011', '0012', '0013', '0014', '0015', '0017', '0016', '0009'}; % ped

for i = 1:length(vid_name)
    make_proposal_flow(strcat(rt_flow, vid_name{i}), strcat(rt_score, vid_name{i}), strcat(rt_output, vid_name{i}));
end
