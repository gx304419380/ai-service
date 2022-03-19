package com.fly.ai.common;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import lombok.experimental.UtilityClass;

import java.awt.image.BufferedImage;

import static java.awt.Image.SCALE_DEFAULT;

/**
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 19:29
 */
@UtilityClass
public class ImageUtils {

    /**
     * 旋转图片90度
     *
     * @param image 图片
     * @return 旋转90度后的图片
     */
    public static Image rotateImage(Image image) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
            return ImageFactory.getInstance().fromNDArray(rotated);
        }
    }



    /**
     * 图片缩放
     *
     * @param original  原始
     * @param width     宽
     * @param height    高
     * @return          缩放后的图片
     */
    public static BufferedImage scale(BufferedImage original, int width, int height) {
        if (width == original.getWidth() && height == original.getHeight()) {
            return original;
        }

        java.awt.Image scaledInstance = original.getScaledInstance(width, height, SCALE_DEFAULT);
        BufferedImage scaledImage = new BufferedImage(width, height, original.getType());
        scaledImage.getGraphics().drawImage(scaledInstance, 0, 0, null);
        return scaledImage;
    }

    /**
     * 图片缩放
     *
     * @param original  原始
     * @param width     宽
     * @param height    高
     * @return          缩放后的图片
     */
    public static Image scale(Image original, int width, int height) {
        if (width == original.getWidth() && height == original.getHeight()) {
            return original;
        }

        try(NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray = original.toNDArray(manager);
            NDArray resize = NDImageUtils.resize(ndArray, width, height);
            resize = resize.toType(DataType.INT8, false);
            return ImageFactory.getInstance().fromNDArray(resize);
        }
    }



}
