package com.fly.ai.yolov8;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static com.fly.ai.common.Constants.ENGINE_ONNX;
import static com.fly.ai.common.ImageUtils.drawDetections;
import static com.fly.ai.common.ImageUtils.scale;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;
import static java.util.Objects.nonNull;

/**
 * @author guoxiang
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2023/4/19 17:36
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class FireService {

    private final FireProperties config;

    private ZooModel<Image, DetectedObjects> yoloModel;

    @PostConstruct
    public void init() throws ModelNotFoundException, MalformedModelException, IOException {
        log.info("开始加载YOLOv8模型");

        Device device = Device.Type.CPU.equalsIgnoreCase(config.getDeviceType()) ? Device.cpu() : Device.gpu();

        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("width", config.getWidth());
        arguments.put("height", config.getHeight());
        arguments.put("resize", true);
        arguments.put("rescale", true);

        //这里我只有两个种类，火焰和烟雾，如果你有多个种类可以使用names文件，使用类似于yolov5的写法optSynsetArtifactName指定文件
        YoloV8RelativeTranslator translator = YoloV8RelativeTranslator
                .builder(arguments)
                .optThreshold(config.getThreshold())
                // .optSynsetArtifactName(yolo.getNameList())
                .optSynset(List.of("火焰", "烟雾"))
                .build();

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optDevice(device)
                .optModelUrls(getRealUrl(config.getModelUrl()))
                .optModelName(config.getModelName())
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .optEngine(config.getEngine())
                .build();

        yoloModel = ModelZoo.loadModel(criteria);

        log.info("YOLO_FIRE模型加载完成");
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
        Image img = ImageFactory.getInstance().fromImage(image);
        return detect(img);
    }


    /**
     * 对象检测函数
     *
     * @param image 图片
     */
    @SneakyThrows
    public DetectedObjects detect(Image image) {

        long startTime = System.currentTimeMillis();

        //开始检测图片
        DetectedObjects detections;
        try (Predictor<Image, DetectedObjects> predictor = yoloModel.newPredictor()) {
            detections = predictor.predict(image);
        }
        log.info("results: {}", detections);

        log.info("detect cost {}ms", System.currentTimeMillis() - startTime);

        return detections;
    }


    /**
     * 检测并绘制结果
     *
     * @param image 原始图片
     * @return 带有绘制结果的图片
     */
    public BufferedImage getResultImage(BufferedImage image) {
        //将图片大小设置为网络输入要求的大小
        BufferedImage scale = scale(image, config.getWidth(), config.getHeight());

        DetectedObjects detections = detect(scale);

        //将结果绘制到图片中
        return drawDetections(image, detections);
    }

}
