import math
import random

import torch
from torchvision.transforms import functional as TF
import numpy as np
import cv2
import imutils

from models import reinforcement_net
import utils
from constants import (
    ARMS_POS_PIXEL,
    ARM_RADIUS_LIMIT_PIXEL,
    GRIPPER_PUSH_RADIUS_PIXEL,
    IMAGE_PAD_SIZE,
    PIXEL_SIZE,
    PUSH_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    NUM_ROTATION,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    BG_THRESHOLD,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
)


class MCTSHelper:
    """
    Simulate the state after push actions.
    Evaluation the grasp rewards.
    """

    def __init__(self, env, grasp_model_path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Initialize Grasp Q Evaluation
        self.grasp_model = reinforcement_net()
        self.grasp_model.load_state_dict(torch.load(grasp_model_path)["model"])
        self.grasp_model = self.grasp_model.to(self.device)
        self.grasp_model.eval()

        self.env = env
        self.action_recorder = {}
        self.state_recorder = {}

    def reset(self):
        self.action_recorder = {}
        self.state_recorder = {}

    def sample_push_actions(
        self, object_states, arm_idx=None, color_image=None, mask_image=None, plot=False
    ):
        """
        Sample actions around the objects, from the boundary to the center.
        Assume there is no object in "black"
        Output the rotated image, such that the push action is from left to right
        """

        # Retrieve information
        if color_image is None:
            self.env.restore_objects(object_states)
            color_image, _, mask_image = utils.get_true_heightmap(self.env)

        # Process mask into binary format
        masks = []
        for mask_id in np.unique(mask_image):
            if mask_id in self.env.obj_ids["rigid"]:
                mask = np.where(mask_image == mask_id, 255, 0).astype(np.uint8)
                masks.append(mask)
        if len(masks) == 0:
            return []

        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.uint8)
        if plot:
            plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        blurred = cv2.medianBlur(gray, 5)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        # find the contour of a single object
        points_on_contour = []
        points = []
        four_idx = []
        other_idx = []
        for oi in range(len(masks)):
            obj_cnt = cv2.findContours(masks[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            obj_cnt = imutils.grab_contours(obj_cnt)
            obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
            if len(obj_cnt) == 0:
                continue
            else:
                obj_cnt = obj_cnt[-1]
                # if too small, then, we skip
                if cv2.contourArea(obj_cnt) < 10:
                    continue
            # get center
            M = cv2.moments(obj_cnt)
            cX = round(M["m10"] / M["m00"])
            cY = round(M["m01"] / M["m00"])
            if plot:
                cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
            # get pca angle
            angle = get_orientation(obj_cnt)
            # get contour points
            skip_num = len(obj_cnt) // 12  # 12 possible pushes for an object
            skip_count = 0
            diff_angle_limit_four = 0.3
            target_diff_angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])
            # add four directions to center of object
            four_poses = [
                (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
                (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
                (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
                (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
            ]
            for pose in four_poses:
                x = pose[0]
                y = pose[1]
                diff_x = cX - x
                diff_y = cY - y
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_on_contour = (round(x), round(y))
                diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
                point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
                points_on_contour.append(point_on_contour)
                points.append(point)
                four_idx.append(len(points) - 1)
            # for pi, p in enumerate(obj_cnt):
            #     x = p[0][0]
            #     y = p[0][1]
            #     if x == cX or y == cY:
            #         continue
            #     diff_x = cX - x
            #     diff_y = cY - y
            #     test_angle = math.atan2(diff_y, diff_x)
            #     should_append = False
            #     # avoid four directions to center of object
            #     if (
            #         np.min(np.abs(abs(angle - test_angle) - target_diff_angles))
            #         < diff_angle_limit_four
            #     ):
            #         should_append = False
            #         skip_count = 0
            #     elif skip_count == skip_num:
            #         should_append = True
            #     if should_append:
            #         diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            #         diff_x /= diff_norm
            #         diff_y /= diff_norm
            #         point_on_contour = (round(x), round(y))
            #         diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
            #         point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
            #         points_on_contour.append(point_on_contour)
            #         points.append(point)
            #         other_idx.append(len(points) - 1)
            #         skip_count = 0
            #     else:
            #         skip_count += 1
        # random actions, adding priority points at the end
        random.shuffle(four_idx)
        random.shuffle(other_idx)
        new_points = []
        new_points_on_contour = []
        for idx in other_idx:
            new_points.append(points[idx])
            new_points_on_contour.append(points_on_contour[idx])
        for idx in four_idx:
            new_points.append(points[idx])
            new_points_on_contour.append(points_on_contour[idx])
        points = new_points
        points_on_contour = new_points_on_contour

        if plot:
            # loop over the contours
            for c in cnts:
                cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

        actions = []
        for pi in range(len(points)):
            # arm limitation
            if arm_idx is not None:
                diff_x = points_on_contour[pi][0] - points[pi][0]
                diff_y = points_on_contour[pi][1] - points[pi][1]
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_to = (
                    int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE),
                    int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE),
                )
            # if arm_idx is not None and (
            #     _distance(ARMS_POS_PIXEL[arm_idx], points[pi]) > ARM_RADIUS_LIMIT_PIXEL
            #     or _distance(ARMS_POS_PIXEL[arm_idx], point_to) > ARM_RADIUS_LIMIT_PIXEL
            # ):
            if arm_idx is not None and (
                abs(ARMS_POS_PIXEL[arm_idx][1] - points[pi][1]) > ARM_RADIUS_LIMIT_PIXEL
                or abs(ARMS_POS_PIXEL[arm_idx][1] - point_to[1]) > ARM_RADIUS_LIMIT_PIXEL
            ):
                qualify = False
            # clearance larger
            elif (
                np.sum(
                    thresh[
                        max(0, points[pi][1] - GRIPPER_PUSH_RADIUS_SAFE_PIXEL) : min(
                            IMAGE_SIZE, points[pi][1] + GRIPPER_PUSH_RADIUS_SAFE_PIXEL + 1
                        ),
                        max(0, points[pi][0] - GRIPPER_PUSH_RADIUS_SAFE_PIXEL) : min(
                            IMAGE_SIZE, points[pi][0] + GRIPPER_PUSH_RADIUS_SAFE_PIXEL + 1
                        ),
                    ]
                    > 0
                )
                == 0
            ):
                qualify = True
            # clearance smaller
            else:
                # compute rotation angle
                down = (0, 1)
                current = (
                    points_on_contour[pi][0] - points[pi][0],
                    points_on_contour[pi][1] - points[pi][1],
                )
                dot = (
                    down[0] * current[0] + down[1] * current[1]
                )  # dot product between [x1, y1] and [x2, y2]
                det = down[0] * current[1] - down[1] * current[0]  # determinant
                angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
                angle = math.degrees(angle)
                crop = thresh[
                    points[pi][1]
                    - GRIPPER_PUSH_RADIUS_SAFE_PIXEL : points[pi][1]
                    + GRIPPER_PUSH_RADIUS_SAFE_PIXEL
                    + 1,
                    points[pi][0]
                    - GRIPPER_PUSH_RADIUS_SAFE_PIXEL : points[pi][0]
                    + GRIPPER_PUSH_RADIUS_SAFE_PIXEL
                    + 1,
                ]
                if crop.shape == (
                    GRIPPER_PUSH_RADIUS_SAFE_PIXEL * 2 + 1,
                    GRIPPER_PUSH_RADIUS_SAFE_PIXEL * 2 + 1,
                ):
                    crop = utils.rotate(crop, angle)
                    (h, w) = crop.shape
                    crop_cy, crop_cx = (h // 2, w // 2)
                    crop = crop[
                        crop_cy
                        - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2) : crop_cy
                        + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
                        + 1,
                        crop_cx
                        - GRIPPER_PUSH_RADIUS_PIXEL : crop_cx
                        + GRIPPER_PUSH_RADIUS_PIXEL
                        + 1,
                    ]
                    qualify = np.sum(crop > 0) == 0
                else:
                    qualify = False
            if qualify:
                if plot:
                    diff_x = points_on_contour[pi][0] - points[pi][0]
                    diff_y = points_on_contour[pi][1] - points[pi][1]
                    diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                    diff_x /= diff_norm
                    diff_y /= diff_norm
                    point_to = (
                        int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
                        int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
                    )
                    if pi < len(other_idx):
                        cv2.arrowedLine(
                            plot_image,
                            points[pi],
                            point_to,
                            (0, 0, 255),
                            2,
                            tipLength=0.2,
                        )
                    else:
                        cv2.arrowedLine(
                            plot_image,
                            points[pi],
                            point_to,
                            (255, 0, 0),
                            2,
                            tipLength=0.2,
                        )
                push_start = (points[pi][1], points[pi][0])
                push_vector = np.array(
                    [
                        points_on_contour[pi][1] - points[pi][1],
                        points_on_contour[pi][0] - points[pi][0],
                    ]
                )
                unit_push = push_vector / np.linalg.norm(push_vector)
                push_end = (
                    round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE),
                    round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE),
                )
                actions.append(
                    PushAction((push_start[0], push_start[1], push_end[0], push_end[1]), arm_idx)
                )

        if plot:
            cv2.imwrite("test_sample_push.png", plot_image)

        return actions

    def get_grasp_action(
        self,
        max_q,
        arm_idx,
        color_image,
        depth_image,
        mask,
        post_checking=True,
        plot=False,
    ):

        # Sample one grasp per object
        best_pix_ind = None
        grasp_q_value, _, grasp_predictions = self.get_grasp_q(
            color_image, depth_image, arm_idx, post_checking
        )
        if np.max(grasp_q_value) > max_q:
            # post process, only grasp one object, focus on target object
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            # focus on target
            for rotate_idx in range(len(grasp_predictions)):
                grasp_predictions[rotate_idx][mask != 255] = 0
            # only grasp one object
            kernel_big = np.ones(
                (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
            )
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = (
                GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
            )
            for rotate_idx in range(len(grasp_predictions)):
                color_mask = utils.rotate(mask, rotate_idx * (360.0 / NUM_ROTATION), True)
                color_mask[color_mask == 0] = 1
                color_mask[color_mask == 255] = 0
                no_target_mask = color_mask
                bg_mask = utils.rotate(mask_bg, rotate_idx * (360.0 / NUM_ROTATION), True)
                no_target_mask[bg_mask == 255] = 0
                # only grasp one object
                invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
                invalid_mask = utils.rotate(
                    invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
                )
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                    grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
                )
                grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)

        if plot:
            plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            rotate_idx = action.pose[0]
            y, x = action.pose[1:3]
            draw_points = np.array(
                [
                    [
                        [
                            x - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                            y - GRIPPER_GRASP_WIDTH_PIXEL // 2,
                        ]
                    ],
                    [
                        [
                            x + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                            y - GRIPPER_GRASP_WIDTH_PIXEL // 2,
                        ]
                    ],
                    [
                        [
                            x + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                            y + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                        ]
                    ],
                    [
                        [
                            x - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                            y + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                        ]
                    ],
                ]
            )
            M = cv2.getRotationMatrix2D((x, y), rotate_idx * (360.0 / NUM_ROTATION), 1)
            draw_points = cv2.transform(draw_points, M)
            cv2.polylines(plot_image, [draw_points], True, (100, 255, 0), 2)
            cv2.imwrite("test_sample_grasp.png", plot_image)

        return best_pix_ind, grasp_predictions, np.max(grasp_predictions)

    def sample_grasp_actions(
        self,
        object_states,
        max_q,
        arm_idx=None,
        color_image=None,
        depth_image=None,
        mask_image=None,
        post_checking=True,
        plot=False,
    ):
        # Retrieve information
        if color_image is None:
            self.env.restore_objects(object_states)
            color_image, depth_image, mask_image = utils.get_true_heightmap(self.env)

        # Sample one grasp per object
        actions = []
        grasp_q_value, _, grasp_predictions = self.get_grasp_q(
            color_image, depth_image, arm_idx, post_checking
        )
        if np.max(grasp_q_value) > max_q:

            for mask_id in np.unique(mask_image):
                if mask_id in self.env.obj_ids["rigid"]:
                    mask = np.where(mask_image == mask_id, 255, 0).astype(np.uint8)

                    # # post process, only grasp one object, focus on blue object
                    # temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
                    # mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
                    # mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
                    # mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
                    # mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
                    # # focus on blue
                    # for rotate_idx in range(len(grasp_predictions)):
                    #     grasp_predictions[rotate_idx][mask_pad != 255] = 0
                    # padding_width_start = IMAGE_PAD_WIDTH
                    # padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
                    # # only grasp one object
                    # kernel_big = np.ones(
                    #     (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
                    # )
                    # if (
                    #     IS_REAL
                    # ):  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
                    #     threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 7
                    #     threshold_small = (
                    #         GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 15
                    #     )
                    # else:
                    #     threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
                    #     threshold_small = (
                    #         GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
                    #     )
                    # depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
                    # for rotate_idx in range(len(grasp_predictions)):
                    #     color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                    #     color_mask[color_mask == 0] = 1
                    #     color_mask[color_mask == 255] = 0
                    #     no_target_mask = color_mask
                    #     bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                    #     no_target_mask[bg_mask == 255] = 0
                    #     # only grasp one object
                    #     invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
                    #     invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
                    #     grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                    #         grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
                    #     )
                    #     grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

                    valid_pixel = grasp_predictions[:, mask == 255]
                    if valid_pixel.size > 0 and np.max(valid_pixel) > max_q:
                        rotate_idx, _ = np.unravel_index(np.argmax(valid_pixel), valid_pixel.shape)
                        valid_mask = cv2.bitwise_and(
                            grasp_predictions[rotate_idx], grasp_predictions[rotate_idx], mask=mask
                        )
                        best_pix_ind = np.unravel_index(np.argmax(valid_mask), valid_mask.shape)
                        actions.append(
                            GraspAction(
                                (rotate_idx, best_pix_ind[0], best_pix_ind[1]), arm_idx, mask_id
                            )
                        )

        if plot:
            plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            for action in actions:
                rotate_idx = action.pose[0]
                y, x = action.pose[1:3]
                draw_points = np.array(
                    [
                        [
                            [
                                x - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                                y - GRIPPER_GRASP_WIDTH_PIXEL // 2,
                            ]
                        ],
                        [
                            [
                                x + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                                y - GRIPPER_GRASP_WIDTH_PIXEL // 2,
                            ]
                        ],
                        [
                            [
                                x + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                                y + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                            ]
                        ],
                        [
                            [
                                x - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                                y + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                            ]
                        ],
                    ]
                )
                M = cv2.getRotationMatrix2D((x, y), rotate_idx * (360.0 / NUM_ROTATION), 1)
                draw_points = cv2.transform(draw_points, M)
                cv2.polylines(plot_image, [draw_points], True, (100, 255, 0), 2)
            cv2.imwrite("test_sample_grasp.png", plot_image)

        return actions

    def simulate_push(self, push_pos, restore_states=None):
        """Simulate the push with a flying gripper, return the new states if success
        Now only push at a fixed height (0.01 cm)"""

        if restore_states is not None:
            self.env.restore_objects(restore_states)
        push_start = [
            push_pos[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            push_pos[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            0.01,
        ]
        push_end = [
            push_pos[2] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            push_pos[3] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            0.01,
        ]

        success = self.env.push(push_start, push_end, verbose=False)
        # Check if push was successful
        if not success:
            return None
        self.env.wait_static()
        object_states = self.env.save_objects()

        # Check if all objects (centers) are still in workspace
        for obj in object_states:
            pos = obj[0]
            if abs(pos[0] - self.env.drop_pose[0]) < 0.1:
                continue
            if (
                pos[0] < WORKSPACE_LIMITS[0][0]
                or pos[0] > WORKSPACE_LIMITS[0][1]
                or pos[1] < WORKSPACE_LIMITS[1][0]
                or pos[1] > WORKSPACE_LIMITS[1][1]
            ):
                return None

        color_image, depth_image, mask_image = utils.get_true_heightmap(self.env)

        return object_states, color_image, depth_image, mask_image

    def simulate_grasp(self, grasp_pos, grasp_height, grasp_obj, restore_states=None):
        """"Simulate the grasp with a flying gripper, return the new states if success"""

        if restore_states is not None:
            self.env.restore_objects(restore_states)

        best_rotation_angle = np.deg2rad(grasp_pos[0] * (360.0 / NUM_ROTATION))
        primitive_position = [
            grasp_pos[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            grasp_pos[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            grasp_height + WORKSPACE_LIMITS[2][0],
        ]

        success, pick_obj = self.env.grasp(primitive_position, best_rotation_angle, verbose=False)
        # Check if grasp was successful and the grasped object matched
        if success and pick_obj == grasp_obj:
            self.env.wait_static()
            object_states = self.env.save_objects()
        else:
            return None

        color_image, depth_image, mask_image = utils.get_true_heightmap(self.env)

        return object_states, color_image, depth_image, mask_image

    @torch.no_grad()
    def get_grasp_q(
        self,
        color_heightmap,
        depth_heightmap,
        arm_idx=None,
        post_checking=True,
    ):
        """Get the grasp reward. The proposed grasp action may grasp more than one object"""
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(
            depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )

        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )

        # Pass input data through model
        output_prob = self.grasp_model(input_color_data, input_depth_data, True, -1, False)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1]
                    .cpu()
                    .data.numpy()[
                        :,
                        0,
                        :,
                        :,
                    ]
                )
            else:
                grasp_predictions = np.concatenate(
                    (
                        grasp_predictions,
                        output_prob[rotate_idx][1]
                        .cpu()
                        .data.numpy()[
                            :,
                            0,
                            :,
                            :,
                        ],
                    ),
                    axis=0,
                )

        # # post process, only grasp one object, focus on target object
        # temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        # mask = cv2.inRange(temp, color_lower, color_upper)
        # mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        # mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        # mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # # focus on target
        # for rotate_idx in range(len(grasp_predictions)):
        #     grasp_predictions[rotate_idx][mask_pad != 255] = 0
        # # only grasp one object
        # kernel_big = np.ones(
        #     (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
        # )
        # threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
        # threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
        # depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
        # for rotate_idx in range(len(grasp_predictions)):
        #     color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
        #     color_mask[color_mask == 0] = 1
        #     color_mask[color_mask == 255] = 0
        #     no_target_mask = color_mask
        #     bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
        #     no_target_mask[bg_mask == 255] = 0
        #     # only grasp one object
        #     invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
        #     invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
        #     grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
        #         grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
        #     )
        #     grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, assuming no overlapping objects.
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            # TODO: if this is not working, we may change GRIPPER_GRASP_WIDTH_PIXEL to GRIPPER_GRASP_SAFE_WIDTH_PIXEL
            check_kernel = np.ones(
                (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
            )
            left_bound = math.floor(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            right_bound = (
                math.ceil(
                    (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
                )
                + 1
            )
            check_kernel[:, left_bound:right_bound] = 0
            for rotate_idx in range(len(grasp_predictions)):
                object_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
                invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
                invalid_mask[invalid_mask > 0] = 255
                invalid_mask = utils.rotate(
                    invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
                )
                grasp_predictions[rotate_idx][invalid_mask == 255] = 0
        grasp_predictions = grasp_predictions[
            :,
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        ]

        # Based on the arm_idx, remove impossible actions
        if arm_idx is not None:
            if ARMS_POS_PIXEL[arm_idx][1] > IMAGE_SIZE // 2:
                y_limit = ARMS_POS_PIXEL[arm_idx][1] - ARM_RADIUS_LIMIT_PIXEL
                for rotate_idx in range(len(grasp_predictions)):
                    grasp_predictions[rotate_idx][:y_limit, :] = 0
            else:
                y_limit = ARM_RADIUS_LIMIT_PIXEL + ARMS_POS_PIXEL[arm_idx][1]
                for rotate_idx in range(len(grasp_predictions)):
                    grasp_predictions[rotate_idx][y_limit:, :] = 0
            # mask = np.zeros_like(grasp_predictions[0], dtype=np.uint8)
            # cv2.circle(mask, ARMS_POS_PIXEL[arm_idx], ARM_RADIUS_LIMIT_PIXEL, 255, -1)
            # for rotate_idx in range(len(grasp_predictions)):
            #     grasp_predictions[rotate_idx][mask != 255] = 0

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return grasp_q_value, best_pix_ind, grasp_predictions

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, is_push=False):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations / 4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row * 4 + canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap(
                    (prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(
                        prediction_vis,
                        (int(best_pix_ind[2]), int(best_pix_ind[1])),
                        7,
                        (0, 0, 255),
                        2,
                    )
                prediction_vis = utils.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations))
                if rotate_idx == best_pix_ind[0]:
                    center = np.array([[[int(best_pix_ind[2]), int(best_pix_ind[1])]]])
                    M = cv2.getRotationMatrix2D(
                        (
                            prediction_vis.shape[1] // 2,
                            prediction_vis.shape[0] // 2,
                        ),
                        rotate_idx * (360.0 / num_rotations),
                        1,
                    )
                    center = cv2.transform(center, M)
                    center = np.transpose(center[0])
                    if is_push:
                        point_from = (int(center[0]), int(center[1]))
                        point_to = (int(center[0] + PUSH_DISTANCE_PIXEL), int(center[1]))
                        prediction_vis = cv2.arrowedLine(
                            prediction_vis,
                            point_from,
                            point_to,
                            (100, 255, 0),
                            2,
                            tipLength=0.2,
                        )
                    else:
                        prediction_vis = cv2.rectangle(
                            prediction_vis,
                            (
                                max(0, int(center[0]) - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2),
                                max(0, int(center[1]) - GRIPPER_GRASP_WIDTH_PIXEL // 2),
                            ),
                            (
                                min(
                                    prediction_vis.shape[1],
                                    int(center[0]) + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2,
                                ),
                                min(
                                    prediction_vis.shape[0],
                                    int(center[1]) + GRIPPER_GRASP_WIDTH_PIXEL // 2,
                                ),
                            ),
                            (100, 255, 0),
                            1,
                        )
                        prediction_vis = cv2.rectangle(
                            prediction_vis,
                            (
                                max(0, int(center[0]) - GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2),
                                max(0, int(center[1]) - GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2),
                            ),
                            (
                                min(
                                    prediction_vis.shape[1],
                                    int(center[0]) + GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2,
                                ),
                                min(
                                    prediction_vis.shape[0],
                                    int(center[1]) + GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2,
                                ),
                            ),
                            (100, 100, 155),
                            1,
                        )
                background_image = utils.rotate(
                    color_heightmap, rotate_idx * (360.0 / num_rotations)
                )
                prediction_vis = (
                    0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis
                ).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas
