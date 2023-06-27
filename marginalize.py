#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 02:27:41 2023

@author: dev
"""

import numpy as np
import cv2
import pandas as pd

# def area(img, start_pos, bw=False):
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     pixels_to_check = [start_pos]

#     if not bw:
#         color = img[start_pos[0], start_pos[1]].tolist()

#         while pixels_to_check:
#             x, y = pixels_to_check.pop(0)

#             if img[x, y].tolist() == color and mask[x, y] == 0:
#                 mask[x, y] = 255

#                 indices = np.clip([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)], 0, np.array(img.shape[:2]) - 1)
#                 valid_indices = np.where(mask[indices[:, 0], indices[:, 1]] == 0)[0]
#                 pixels_to_check.extend(indices[valid_indices])

#     else:
#         color = img[start_pos[0], start_pos[1]]

#         while pixels_to_check:
#             x, y = pixels_to_check.pop(0)

#             if img[x, y] == color and mask[x, y] == 0:
#                 mask[x, y] = 255

#                 indices = np.clip([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)], 0, np.array(img.shape[:2]) - 1)
#                 valid_indices = np.where(mask[indices[:, 0], indices[:, 1]] == 0)[0]
#                 pixels_to_check.extend(indices[valid_indices])

#     ar = np.count_nonzero(mask)
#     return mask, ar


def area(img, start_pos, bw=False):

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pixels_to_check = [start_pos]

    if not bw:
        color = tuple(img[start_pos[0], start_pos[1]].tolist())

        while pixels_to_check:
            x, y = pixels_to_check.pop(0)

            if tuple(img[x, y].tolist()) == color and mask[x, y] == 0:
                mask[x, y] = 255

                if x > 0:
                    pixels_to_check.append((x - 1, y))
                if y > 0:
                    pixels_to_check.append((x, y - 1))
                if x < img.shape[1] - 1:
                    pixels_to_check.append((x + 1, y))
                if y < img.shape[0] - 1:
                    pixels_to_check.append((x, y + 1))
    else:
        color = img[start_pos[0], start_pos[1]]

        while pixels_to_check:
            x, y = pixels_to_check.pop(0)

            if img[x, y] == color and mask[x, y] == 0:
                mask[x, y] = 255

                if x > 0:
                    pixels_to_check.append((x - 1, y))
                if y > 0:
                    pixels_to_check.append((x, y - 1))
                if x < img.shape[1] - 1:
                    pixels_to_check.append((x + 1, y))
                if y < img.shape[0] - 1:
                    pixels_to_check.append((x, y + 1))


    ar = np.count_nonzero(mask)

    return mask, ar


def boundary(mask, flag = False, kern = 3):
    kernel = np.ones((kern, kern), np.uint8)
    img_dilation = cv2.dilate(mask, kernel, iterations=1)
    boundary_mask = img_dilation-mask
    if flag:
        return np.sum(boundary_mask > 0)
    else:
        return boundary_mask


def count_pixels_within_boundary(img, mask):
    boundary_mask = boundary(mask)

    pixels = img[boundary_mask != 0]

    unique_colors, counts = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0, return_counts=True)

    if (boundary_mask*mask).max():
        raise Exception('Dilation inappropriate')

    return dict(zip(map(tuple, unique_colors), counts))


def largest_boundary(img, mask):
    color_counts = count_pixels_within_boundary(img, mask)

    # print('color counts: ', color_counts)
    if color_counts:
        max_boundary_color = None
        max_boundary_length = 0

        for color, count in color_counts.items():
            boundary_length = count
            # print(f'boundary_length: {boundary_length} max_boundary_length {max_boundary_length} color {color}')

            if boundary_length > max_boundary_length:
                max_boundary_length = boundary_length
                max_boundary_color = color

        return max_boundary_color

    else:
        return None

def fill_color(img, start_pos, target_color, mask='hello'):
    if isinstance(mask, str):
        mask, _ = area(img, start_pos)
        color = tuple(img[start_pos[0], start_pos[1]].tolist())
    else:
        color = (1, 1, 1)
    try:
        target_color = target_color.tolist()
    except:
        pass

    if target_color != color:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i, j]:
                    img[i, j] = target_color
    else:
        # This piece of code will find a new target color if the supplied target color was inappropriate
        break_outer_loop = False
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if tuple(img[i, j].tolist()) != target_color:
                    # print(tuple(img[i, j].tolist()))
                    # print(target_color)
                    target_color = tuple(img[i, j].tolist())
                    break_outer_loop = True
                    break
            if break_outer_loop:
                break

        # print('new_target_color', target_color)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i, j]:
                    img[i, j] = target_color
    return img


def remove_noise(img, masks):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            start_pos = [i, j]
            if max([m[i, j] for m in masks]):
                continue
            else:
                mask, ar = area(img, start_pos)
                # if ar <= 5 and not mask[img.shape[0]//2, img.shape[1]//2]:
                if ar <= 1:
                    img = fill_color(img, start_pos, tuple(img[i, j-1].tolist()), mask)
                masks.append(mask)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            start_pos = [i, j]
            if max([m[i, j] for m in masks]):
                continue
            else:
                mask, ar = area(img, start_pos)
                # if ar <= 5 and not mask[img.shape[0]//2, img.shape[1]//2]:
                if ar <= 5:
                    c = largest_boundary(img, start_pos, mask)
                    if c:
                        img = fill_color(img, start_pos, c, mask)
                    else:
                        img = fill_color(img, start_pos, tuple(img[i, j-1].tolist()), mask)
                masks.append(mask)
    return img, masks


def remove_edge_artefacts(img):
    r1 = range(1, 21)
    r2 = range(222, 202, -1)
    for i, j in zip(r1, r2):
        c1 = min(img[i, i] == img[i-1, i-1])
        c2 = min(img[i, j] == img[i-1, j+1])
        c3 = min(img[j, i] == img[j+1, i-1])
        c4 = min(img[j, j] == img[j+1, j+1])
        # print('[i, k]', [i, j])
        # print('c1: ', c1, 'c2: ', c2, 'c3: ', c3, 'c4: ', c4)
        if (not c3 and not c4):
        # or max(img[j, 112] != img[j+1, 112]):
            # print('img[i, 112] != img[i-1, 112]', img[i, 112] != img[i-1, 112])
            # print('lower edge changed')
            img = fill_color(img, [j+1, 112], img[j, 112])
        elif (not c1 and not c3):
        # or max(img[112, i-1] != img[112, i]):
            # print('left edge changed')
            img = fill_color(img, [112, i-1], img[112, i])
        elif (not c1 and not c2):
        # or max(img[i-1, 112] != img[i, 112]):
            # print('upper edge changed')
            img = fill_color(img, [i-1, 112], img[i, 112])
        elif (not c2 and not c4):
        # or max(img[112, j+1] != img[112, j]):
            # print('right edge changed')
            img = fill_color(img, [112, j+1], img[112, j])
        elif c1 == c2 == c3 == c4 == 1:
            # print('All edges ok')
            pass

    #if any 2 of them flip at the same time, that was an edge artefact
    # cv2.imshow('edge removal', img)
    # cv2.waitKey(2000)
    return img


def check_num_regions(img, bw=False):
    masks = [np.zeros((224, 224), dtype=np.uint8)]
    for i in range(224):
        for j in range(224):
            start_pos = [i, j]
            if max([m[i, j] for m in masks]) or np.max(img[i, j]) == 0:
                continue
            else:
                mask, _ = area(img, start_pos, bw=bw)
                masks.append(mask)
    return len(masks), masks


def flatten(mask, down=True):
    if down:
        out_mask = (mask == 255)
    else:
        out_mask = np.zeros((224, 224))
        for i in range(224):
            for j in range(224):
                if mask[i, j]:
                    out_mask[i, j] = 255
    return out_mask


def remove_enclosure(img):
    num_regions, final_masks = check_num_regions(img)
    for msk in final_masks[1:]:
        for msk2 in final_masks[1:]:
            if (flatten(flatten(boundary(msk)) & flatten(msk2), False) == flatten(boundary(msk), False)).all():
                # msk2 contains msk
                flag = 1
                for i in range(224):
                    for j in range(224):
                        if msk2[i, j]:
                            start_pos = [i, j]
                            flag = 0
                            break
                    if not flag:
                        break

                img = fill_color(img, start_pos, [0, 0, 0])
    return img


def marginalize(img, thresh = 0.001):
    mask = np.zeros((224, 224), dtype=np.uint8)
    masks = [mask]

    if thresh == 0.001:
        img, _ = remove_noise(img, masks)
        print('Noise removed')
        prev_img = np.zeros((224, 224, 3), dtype=np.uint8)
        while np.max(prev_img != img):
            prev_img = img
            img = remove_edge_artefacts(img)



    else:
        img = remove_edge_artefacts(img)
        threshold = thresh*224*224

        flag = 1
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                start_pos = [i, j]
                buf = img
                if max([m[i, j] for m in masks]) or i == 112 or j == 112:
                    continue
                else:
                    mask, ar = area(img, start_pos)
                    if ar <= threshold:
                        print('ar: ', ar)
                        c = largest_boundary(img, mask)
                        if c:
                            img = fill_color(img, start_pos, c, mask)
                        else:
                            raise Exception('Inaccurate boundary at pos: ', start_pos)
                        # x = input()
                    masks.append(mask)
                if ar < 15:
                    continue
                else:
                    num_regions, final_masks = check_num_regions(img)
                    if num_regions < 3:
                        img = buf
                        flag = 0
                        break

            if not flag:
                break


    # if thresh == 0.032:
    #     pixels = img.reshape(-1, img.shape[-1])
    #     unique_colors = np.unique(pixels, axis=0)
    #     prev_img = np.zeros((224, 224, 3), dtype=np.uint8)
    #     while len(unique_colors) > 2 and np.max(prev_img != img):
    #         prev_img = img
    #         print('entered_marginalize while loop')
    #         img = remove_enclosure(img)
    #         pixels = img.reshape(-1, img.shape[-1])
    #         unique_colors = np.unique(pixels, axis=0)
    return img

'''TRY TO KNOW SHAPE OF MASK, THAT WILL HELP YOU KNOW which locations to focus on (center/circular edge/extreme edge);
IF A REGION IS ENCLOSED INSIDE ANOTHER REGION, MEANS THE OUTER REGION IS BACKGROUND, which means, if a region's boundary coincides totally with another region's mask, the mask owner is boundary, turn it black'''