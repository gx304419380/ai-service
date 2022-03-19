package com.fly.ai.yolo;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static com.fly.ai.common.Constants.ENGINE_ONNX;
import static com.fly.ai.common.ImageUtils.drawDetections;
import static com.fly.ai.common.ImageUtils.scale;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;
import static java.util.Objects.nonNull;

@Slf4j
@Component
@RequiredArgsConstructor
public class YoloUtils {

    private final YoloProperties yolo;

    private ZooModel<Image, DetectedObjects> yoloModel;

    @PostConstruct
    public void init() throws ModelNotFoundException, MalformedModelException, IOException {
        log.info("开始加载YOLO模型");

        Device device = Device.Type.CPU.equalsIgnoreCase(yolo.getDeviceType()) ? Device.cpu() : Device.gpu();

        Translator<Image, DetectedObjects> translator = YoloV5Translator
                .builder()
                .optThreshold(yolo.getThreshold())
                .optSynsetArtifactName(yolo.getNameList())
                .build();

        YoloV5RelativeTranslator myTranslator = new YoloV5RelativeTranslator(translator, yolo);

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optDevice(device)
                .optModelUrls(getRealUrl(yolo.getYoloUrl()))
                .optModelName(yolo.getModelName())
                .optTranslator(myTranslator)
                .optEngine(ENGINE_ONNX)
                .build();

        yoloModel = ModelZoo.loadModel(criteria);
    }

    @PreDestroy
    public void destroy() {
        if (nonNull(yoloModel)) {
            yoloModel.close();
        }

        log.info("yolo model closed...");
    }


    /**
     * 对象检测函数
     *
     * @param image 图片，尺寸需满足yolo网络入参大小
     */
    @SneakyThrows
    public DetectedObjects detect(BufferedImage image) {
        final BufferedImage scale = scale(image, yolo.getWidth(), yolo.getHeight());
        Image img = ImageFactory.getInstance().fromImage(scale);
        return detect(img);
    }


    /**
     * 对象检测函数
     *
     * @param image 图片
     */
    @SneakyThrows
    public DetectedObjects detect(Image image) {
        Image scaledImage = scale(image, yolo.getWidth(), yolo.getHeight());

        long startTime = System.currentTimeMillis();

        //开始检测图片
        DetectedObjects detections;
        try(Predictor<Image, DetectedObjects> predictor = yoloModel.newPredictor()) {
            detections = predictor.predict(scaledImage);
        }
        log.info("results: {}", detections);

        log.info("detect cost {}ms", System.currentTimeMillis() - startTime);

        return detections;
    }


    /**
     * 检测并绘制结果
     *
     * @param image 原始图片
     * @return      带有绘制结果的图片
     */
    public BufferedImage getResultImage(BufferedImage image) {
        //将图片大小设置为网络输入要求的大小
        BufferedImage scale = scale(image, yolo.getWidth(), yolo.getHeight());

        DetectedObjects detections = detect(scale);

        //将结果绘制到图片中
        return drawDetections(image, detections);
    }

}
