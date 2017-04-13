function make_proposal_flow(flow_root, score_root, output_root)

    %temporal_window_size = 7;
    %half_tws = floor(temporal_window_size / 2);


    % mkdir_if_missing(output_root);
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end

    disp(score_root);

    %% optical flow
   
    frame_name = dir(fullfile(score_root, '*.mat'));
    disp(frame_name);
    n_frame = length(frame_name);
    
    frame = struct('boxes',[],'zs',[]);
    %frame = struct('boxes',[]);
    frame(n_frame).boxes = [];
    
    fprintf(' Loading boxes.');
    for frame_idx = 1:n_frame
        frame_idx
        file_name = fullfile(score_root, frame_name(frame_idx).name);
        dot_pos = findstr(frame_name(frame_idx).name, '.');
        dot_pos = dot_pos(1);
        optflow_name = fullfile(flow_root, [frame_name(frame_idx).name(1:dot_pos-1) '.png']);
        
        frame(frame_idx) = load(file_name);
        if isempty(frame(frame_idx).boxes)
            continue;
        end
        
        %if ~exist(optflow_name, 'file'), continue; end
        
        optflow = imread(optflow_name);
        x_map = single(optflow(:,:,1)) / 255 * 30 - 15;
        y_map = single(optflow(:,:,2)) / 255 * 30 - 15;
        [m,n] = size(x_map);
        box_avg_x = boxes_average_sum(x_map, frame(frame_idx).boxes);
        box_avg_y = boxes_average_sum(y_map, frame(frame_idx).boxes);
        
        boxes = frame(frame_idx).boxes;

        %fprintf('frame_idx: %d, %s\n', frame_idx, file_name)
        
        boxes = boxes + [box_avg_x, box_avg_y, box_avg_x, box_avg_y];
        boxes(:, 1) = max(boxes(:, 1), 1);
        boxes(:, 2) = max(boxes(:, 2), 1);
        boxes(:, 3) = min(boxes(:, 3), n);
        boxes(:, 4) = min(boxes(:, 4), m);
        
        output_dir = output_root;
        mkdir_if_missing(output_dir);
        output_path = fullfile(output_dir, frame_name(frame_idx).name);
        save(output_path, 'boxes');
        clear boxes;
    end
end



function values = boxes_average_sum(map, boxes, box_ratio)
% Author Hongsheng Li

if nargin == 2
    box_ratio = 1.0;
end

[m, n] = size(map);
accum_map = cumsum(cumsum(map,1),2);

col1 = boxes(:,1);
row1 = boxes(:,2);
col2 = boxes(:,3);
row2 = boxes(:,4);

n_row = row2 - row1 + 1;
n_col = col2 - col1 + 1;

col1 = round(col1 + 0.5*(1-box_ratio)*n_col);
row1 = round(row1 + 0.5*(1-box_ratio)*n_row);
col2 = round(col2 - 0.5*(1-box_ratio)*n_col);
row2 = round(row2 - 0.5*(1-box_ratio)*n_row);

col1 = max(1, col1);
row1 = max(1, row1);
col2 = min(n, col2);
row2 = min(m, row2);

n_row = row2 - row1 + 1;
n_col = col2 - col1 + 1;

col1 = col1-1;
row1 = row1-1;
col_out_idx = col1==0;
row_out_idx = row1==0;
corner_out_idx = col_out_idx | row_out_idx;

col1(col_out_idx) = 1;
row1(row_out_idx) = 1;

func = @(x, t)(max(1, min(x, t)));

row1 = func(row1, m);
row2 = func(row2, m);
col1 = func(col1, n);
col2 = func(col2, n);

sum_idx = sub2ind([m,n],row2, col2);
row_idx = sub2ind([m,n],row1, col2);
col_idx = sub2ind([m,n],row2, col1);
corner_idx = sub2ind([m,n],row1,col1);

sum_values = accum_map(sum_idx);
corner_values = accum_map(corner_idx);
col_values = accum_map(col_idx);
row_values = accum_map(row_idx);

corner_values(corner_out_idx) = 0;
col_values(col_out_idx) = 0;
row_values(row_out_idx) = 0;

values = sum_values - col_values - row_values + corner_values;
values = values ./ (n_row .* n_col);
end

function made = mkdir_if_missing(path)
made = false;
if exist(path) == 0
  unix(['mkdir -p ' path]);
  made = true;
end
end
