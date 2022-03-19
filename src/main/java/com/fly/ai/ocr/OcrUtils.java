package com.fly.ai.ocr;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.paddlepaddle.zoo.cv.imageclassification.PpWordRotateTranslator;
import ai.djl.paddlepaddle.zoo.cv.objectdetection.PpWordDetectionTranslator;
import ai.djl.paddlepaddle.zoo.cv.wordrecognition.PpWordRecognitionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import static com.fly.ai.common.Constants.ENGINE_PADDLE;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;
import static com.fly.ai.ocr.OcrType.QUICK;

/**
 * @author 别动我
 * @since 2022/3/16 17:01
 */
@Component
@Slf4j
public class OcrUtils {



    private ZooModel<Image, DetectedObjects> detectionModel;
    private ZooModel<Image, String> recognitionModel;
    private ZooModel<Image, DetectedObjects> detectionQuickModel;
    private ZooModel<Image, String> recognitionQuickModel;
    private ZooModel<Image, Classifications> rotateModel;

    private final OcrProperties ocrProperties;

    public OcrUtils(OcrProperties ocrProperties) {
        this.ocrProperties = ocrProperties;
    }

    /**
     * 文字识别
     *
     * @param inputStream 输入流
     * @param ocrType     识别类型，QUICK快速识别，PRECISE精确识别
     * @return 结果
     */
    @SneakyThrows
    public DetectedObjects ocr(InputStream inputStream, OcrType ocrType) {
        Image image = ImageFactory.getInstance().fromInputStream(inputStream);
        return ocr(image, ocrType);
    }


    /**
     * 文字识别
     *
     * @param url     url
     * @param ocrType 识别类型，QUICK快速识别，PRECISE精确识别
     * @return 结果
     */
    @SneakyThrows
    public DetectedObjects ocr(String url, OcrType ocrType) {
        Image image = ImageFactory.getInstance().fromUrl(url);
        return ocr(image, ocrType);
    }


    /**
     * 文字识别
     *
     * @param image   图片
     * @param ocrType 识别类型，QUICK快速识别，PRECISE精确识别
     * @return 检测结果
     */
    public DetectedObjects ocr(Image image, OcrType ocrType) {
        ocrType = Optional.ofNullable(ocrType).orElse(QUICK);
        DetectedObjects detect = detect(image, ocrType);

        List<String> names = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> rect = new ArrayList<>();

        List<DetectedObjects.DetectedObject> list = detect.items();
        for (DetectedObjects.DetectedObject result : list) {
            BoundingBox box = result.getBoundingBox();
            Image subImg = getSubImage(image, box);

            if (subImg.getHeight() * 1.0 / subImg.getWidth() > 1.5) {
                subImg = rotateImg(subImg);
            }

            Classifications.Classification classifications = checkRotate(subImg);
            if ("Rotate".equals(classifications.getClassName()) && classifications.getProbability() > ocrProperties.getRotateThreshold()) {
                subImg = rotateImg(subImg);
            }

            String name = recognize(subImg, ocrType);
            names.add(name);
            prob.add(1.0);
            rect.add(box);
        }

        return new DetectedObjects(names, prob, rect);
    }

    /**
     * 保存图片
     *
     * @param image 图片
     * @return image
     */
    public BufferedImage createResultImage(Image image, DetectedObjects result) {
        Image newImage = image.duplicate();
        newImage.drawBoundingBoxes(result);
        return (BufferedImage) newImage.getWrappedImage();
    }

    /**
     * 文字识别
     *
     * @param image   图片
     * @param ocrType 识别类型：快速or精确
     */
    @SneakyThrows
    private String recognize(Image image, OcrType ocrType) {
        try (Predictor<Image, String> recognizer = QUICK.equals(ocrType) ?
                recognitionQuickModel.newPredictor() : recognitionModel.newPredictor()) {
            return recognizer.predict(image);
        }
    }


    /**
     * 判断文字角度，如果需要旋转则进行相应处理
     *
     * @return Classifications
     */
    @SneakyThrows
    private Classifications.Classification checkRotate(Image image) {
        try (Predictor<Image, Classifications> rotateClassifier = rotateModel.newPredictor()) {
            Classifications predict = rotateClassifier.predict(image);
            log.debug("word rotate: {}", predict);
            return predict.best();
        }
    }


    /**
     * 旋转图片
     *
     * @param image 图片
     * @return 旋转90度后的图片
     */
    private Image rotateImg(Image image) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
            return ImageFactory.getInstance().fromNDArray(rotated);
        }
    }


    /**
     * 检测文字所在区域
     *
     * @param image   图片
     * @param ocrType 识别类型，QUICK快速识别，PRECISE精确识别
     * @return 检测结果
     */
    @SneakyThrows
    public DetectedObjects detect(Image image, OcrType ocrType) {

        long startTime = System.currentTimeMillis();

        DetectedObjects result;
        //检测文字所在区域
        try (Predictor<Image, DetectedObjects> detector = QUICK.equals(ocrType) ?
                detectionQuickModel.newPredictor() : detectionModel.newPredictor()) {
            result = detector.predict(image);
        }

        long endTime = System.currentTimeMillis();
        log.debug("检测时长{}mm, 结果：{}", (endTime - startTime), result);

        return result;
    }


    /**
     * 获取文字所在的区块图
     *
     * @param img 原始图片
     * @param box 子区块
     * @return 带有文字的image
     */
    public Image getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
        int width = img.getWidth();
        int height = img.getHeight();
        int[] recovered = {
                (int) (extended[0] * width),
                (int) (extended[1] * height),
                (int) (extended[2] * width),
                (int) (extended[3] * height)
        };
        return img.getSubImage(recovered[0], recovered[1], recovered[2], recovered[3]);
    }

    /**
     * 扩展文字块的大小，因为检测出来的文字块比实际文字块要小
     *
     * @param x      x
     * @param y      y
     * @param width  宽
     * @param height 高
     * @return 扩展后的大小
     */
    public double[] extendRect(double x, double y, double width, double height) {
        double centerX = x + width / 2;
        double centerY = y + height / 2;
        if (width > height) {
            width += height * 2.0;
            height *= 3.0;
        } else {
            height += width * 2.0;
            width *= 3.0;
        }
        double newX = centerX - width / 2 < 0 ? 0 : centerX - width / 2;
        double newY = centerY - height / 2 < 0 ? 0 : centerY - height / 2;
        double newWidth = newX + width > 1 ? 1 - newX : width;
        double newHeight = newY + height > 1 ? 1 - newY : height;
        return new double[]{newX, newY, newWidth, newHeight};
    }

    /**
     * 初始化神经网络模型
     */
    @SneakyThrows
    @PostConstruct
    private void init() {
        log.info("开始加载Ocr工具类");

        Device device = Device.Type.CPU.equalsIgnoreCase(ocrProperties.getDeviceType()) ? Device.cpu() : Device.gpu();

        //加载神经网络模型，文字检测和识别都使用百度的PaddlePaddle
        detectionModel = Criteria.builder()
                .optEngine(ENGINE_PADDLE)
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(getRealUrl(ocrProperties.getDetectUrl()))
                .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<>()))
                .optDevice(device)
                .build()
                .loadModel();
        logModelInfo(detectionModel);

        detectionQuickModel = Criteria.builder()
                .optEngine(ENGINE_PADDLE)
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(getRealUrl(ocrProperties.getDetectQuickUrl()))
                .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<>()))
                .optDevice(device)
                .build()
                .loadModel();
        logModelInfo(detectionQuickModel);

        //加载文字识别神经网络模型
        recognitionModel = Criteria.builder()
                .optEngine(ENGINE_PADDLE)
                .setTypes(Image.class, String.class)
                .optModelUrls(getRealUrl(ocrProperties.getRecognizeUrl()))
                .optTranslator(new PpWordRecognitionTranslator())
                .optDevice(device)
                .build()
                .loadModel();
        logModelInfo(recognitionModel);

        recognitionQuickModel = Criteria.builder()
                .optEngine(ENGINE_PADDLE)
                .setTypes(Image.class, String.class)
                .optModelUrls(getRealUrl(ocrProperties.getRecognizeQuickUrl()))
                .optTranslator(new PpWordRecognitionTranslator())
                .optDevice(device)
                .build()
                .loadModel();
        logModelInfo(recognitionQuickModel);

        //加载文字旋转判别模型
        rotateModel = Criteria.builder()
                .optEngine(ENGINE_PADDLE)
                .setTypes(Image.class, Classifications.class)
                .optModelUrls(getRealUrl(ocrProperties.getRotateUrl()))
                .optTranslator(new PpWordRotateTranslator())
                .optDevice(device)
                .build()
                .loadModel();
        logModelInfo(rotateModel);

        log.info("加载Ocr工具类完成");
    }

    /**
     * 服务停止关闭所有模型
     */
    @PreDestroy
    public void closeAll() {
        detectionModel.close();
        recognitionModel.close();
        detectionQuickModel.close();
        recognitionQuickModel.close();
        rotateModel.close();
        log.info("close all models");
    }




    /**
     * 打印model信息
     *
     * @param model 模型
     */
    private void logModelInfo(ZooModel<?, ?> model) {
        log.info("model name：{}, model path: {}", model.getName(), model.getModelPath());
    }


}
