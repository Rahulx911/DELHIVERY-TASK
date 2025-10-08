# DELHIVERY-TASK
**Task 1 — Parcel Detection & Counting**
**Method** 
Employ two complementary detectors for solid coverage:
Mask R-CNN for case instance masks (well suited for irregular shapes, partial occlusion).
Faster R-CNN for additional box proposals (beneficial if masks are absent).
Include edge-based proposals (Canny → contour boxes) as low-confidence candidates to aid detection of thin / low-texture parcels.
Filter, clip and merge detections (area, confidence thresholds).
Perform NMS to eliminate redundant detections.
Give each remaining detection a unique ID and superimpose ID + box/mask on image. Print total count somewhere conspicuous.

**Algorithms & key steps**
Preprocess: read image → tensor conversion → feed to models.
process_model_outputs: normalize boxes, resize/threshold masks → generate dicts {bbox, mask, score, centroid, source}.
Filtering: eliminate detections too small (MIN_AREA_PX) and too big (MAX_AREA_RATIO * img_area); retain only score >= CONF_THRESH.
safe_nms: execute torchvision nms using NMS_IOU to retain clean set.
Visualization: draw bounding box or overlay semi-transparent colored mask; place ID: # text printed on the parcel (center or top-left of bbox). Also print the total count in header.

**Implementation**
Function: run_inference_single(img_path, mask_model, fast_model)
Edge proposals: simple_edge_detection(img_bgr) with area and aspect-ratio filters.
Post-filter and NMS: MIN_AREA_PX, MAX_AREA_RATIO, CONF_THRESH, NMS_IOU.
Visual overlay: draw_detections() or draw_parcels_with_picking_sequence() with unique colors and printed ID.

**Task 2 — Dimensional & Volumetric Analysis**

**Method**
2D dimension measurement: Employ the axis-aligned or oriented bounding box of each bbox/mask:
length_px = major axis (max of height, width)
breadth_px = minor axis (min of height, width)
Height (depth) estimation — multi-cue fusion:
Heuristic cues blended into one estimate:

Perspective position (centroid y-position): objects lower in image tend to be closer → shift depth.
Vanishing-point distance: distance from an assumed vanishing point affects perceived depth.
Size-based assumption: relative area vs image area — bigger apparent area suggests closer object and probably larger height.
Weight the cues and clamp final depth estimate to reasonable bounds (MIN_HEIGHT_RATIO, MAX_HEIGHT_RATIO) to prevent outlier values.
Return a depth_confidence calculated from box area, edge-distance, and aspect ratio heuristics.
Justification: absent metric calibration or depth sensor, these visual cues are the best stable heuristics from a single 2D image. Fusion mitigates single-cue failure modes (e.g., a small package that is low in image will have balanced estimate).

**Volume estimation:**
volume_px3 = length_px * breadth_px * estimated_depth_px
Also calculate a simple surface-area proxy for grasp planning.

**Algorithms & key steps**
calculate_2d_dimensions(bbox) → returns length, breadth, orientation, area, aspect ratio.
estimate_depth_from_perspective(bbox, img_shape, centroid) → returns estimated_depth_px, depth_confidence, depth_reasoning.
calculate_volume(dimensions_2d, depth_info) → returns volume_px3, surface_area_px2.

**Implementation**
Use mask if present for improved bounding box/oriented box; else use detection bbox.
Return human-readable depth_reasoning (e.g., "Lower position suggests closer; large size provides good depth cues").
Save per-parcel analysis in a structured list/dict (CSV/JSON) for downstream use.


**Task 3 — Occlusion Analysis & Topmost Parcel Identification**

**Method**
Overlap analysis: calculate pairwise overlap between parcels (intersection area divided by the area of the occluding parcel). Gives an overlap matrix overlap[i][j] = intersection_area / area_i.
Depth proxy score: calculate a depth_proxy_score (higher indicates likely higher/up in stack) by aggregating:
Vertical position (centroid y): normalized and flipped so that greater image position → greater score.
Relative size (box area) heuristics.
Detection confidence (score).
View coverage (seen by 2 views → more likely accessible).
Depth-confidence from Task 2 combined when present.
Occlusion metric: calculate occlusion_ratio per parcel by combining:
total_overlap_received (how much other parcels overlap it).
depth_penalty (1 − depth_proxy_score).
occlusion_by_higher_parcels: overlap weighted by how much overlapping parcels have greater depth proxy.
neighborhood_penalty: density of surrounding parcels.
Accessibility score: accessibility_score = 1 − occlusion_ratio.
Topmost selection: the parcel with highest accessibility_score is marked topmost.

**Algorithms & key steps**
calculate_spatial_overlap_matrix(parcels) → returns overlap matrix utilized by occlusion computations.
calculate_depth_proxy_scores(parcels, img_shape, dimensional_data) → returns per-parcel depth proxy breakdown.
calculate_occlusion_ratios(parcels, overlap_matrix, depth_scores) → returns occlusion and accessibility results.
identify_topmost_parcel(.) → returns index/ID of topmost parcel + per-parcel analysis.

**Outputs**
Per-parcel: occlusion_ratio, accessibility_score, depth_proxy_score, spatial_overlap, neighbor_count.
Global: topmost_parcel_id and its accessibility_score.
Full overlap_matrix for diagnostics.

**Task 4 — Picking Sequence Optimization**
**Method**
Construct candidate list from merged_parcels + Task2 + Task3 outputs.
For every parcel calculate:
accessibility_percent = accessibility_score * 100
is_pickable = accessibility_score >= 0.70
vertical_priority = 1 − normalized_y where normalized_y = centroid_y / img_height (higher vertical_priority = higher parcel).
priority_score = ACCESSIBILITY_WEIGHT * accessibility_score + VERTICAL_POSITION_WEIGHT * vertical_priority (only if pickable; otherwise priority = 0).

Categorize statuses:
PRIORITY: >= 95% accessibility
HIGH: 85–95%
PICKABLE: 70–85%
BLOCKED: <70%

Sequence generation:
Order pickable parcels by (priority_score desc, normalized_y asc (higher parcels first))
Add blocked parcels later on (DEFERRED), ordered by accessibility_score desc so robot re-picks them afterwards.

Generate outputs:
Visual annotated bottom image with sequence numbers, colored boxes, a center bubble with sequence number, and status tags.
Printable large-font table image with sequence, coords, est dims, est volume, accessibility, rationale.

**Algorithms & key steps**
calculate_picking_priorities(parcels, dimensional_data, occlusion_analysis, img_shape) → generates candidate metadata.
generate_optimal_picking_sequence(picking_candidates) → sorts and allocates sequence numbers.
create_comprehensive_table(picking_sequence, output_path) → generates large-font table PNG.
draw_parcels_with_picking_sequence(img_bgr, parcels, picking_sequence) → final visualization.

**Key parameters**
MIN_ACCESSIBILITY_THRESHOLD = 0.70
ACCESSIBILITY_WEIGHT = 0.6
VERTICAL_POSITION_WEIGHT = 0.4
Status cutoffs: 95% (PRIORITY), 85% (HIGH), 70% (PICKABLE)
