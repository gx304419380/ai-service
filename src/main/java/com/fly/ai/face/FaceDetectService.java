package com.fly.ai.face;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.fly.ai.common.PredictException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import java.awt.image.BufferedImage;
import java.io.IOException;

import static com.fly.ai.common.Constants.ENGINE_PYTORCH;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;

/**
 * 人脸检测服务
 *
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/20 12:57
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class FaceDetectService {

    private final FaceProperties prop;

    private ZooModel<Image, DetectedObjects> detectModel;

    /**
     * 初始化模型
     */
    @PostConstruct
    public void init() throws ModelNotFoundException, MalformedModelException, IOException {
        double confThresh = prop.getConfThresh();
        double nmsThresh = prop.getNmsThresh();
        int topK = 5000;
        double[] variance = {0.1f, 0.2f};
        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32};
        FaceDetectionTranslator translator =
                new FaceDetectionTranslator(confThresh, nmsThresh, variance, topK, scales, steps);

        Device device = Device.Type.CPU.equalsIgnoreCase(prop.getDeviceType()) ? Device.cpu() : Device.gpu();

        log.info("开始加载人脸检测模型");

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(getRealUrl(prop.getDetectModelUrl()))
                .optDevice(device)
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .optEngine(ENGINE_PYTORCH)
                .build();

        detectModel = criteria.loadModel();

        log.info("人脸检测模型加载完毕");
    }


    @PreDestroy
    public void closeResource() {
        detectModel.close();
        log.info("人脸检测模型关闭");
    }


    /**
     * 人脸检测
     *
     * @param image image
     * @return      检测结果
     */
    public DetectedObjects detect(Image image) {
        DetectedObjects detectedObjects;

        long start = System.currentTimeMillis();
        try(Predictor<Image, DetectedObjects> predictor = detectModel.newPredictor()) {
            detectedObjects = predictor.predict(image);
        } catch (Exception e) {
            throw new PredictException("人脸检测出错", e);
        }

        log.debug("人脸检测耗时{}ms", System.currentTimeMillis() - start);
        return detectedObjects;
    }

    /**
     * 绘制人脸框
     *
     * @param image     image
     */
    public BufferedImage detectAndDraw(BufferedImage image) {
        Image img = ImageFactory.getInstance().fromImage(image);
        DetectedObjects detection = detect(img);
        img.drawBoundingBoxes(detection);

        return (BufferedImage) img.getWrappedImage();
    }
}
