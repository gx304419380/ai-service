package com.fly.ai.yolo;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
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
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import static com.fly.ai.common.Constants.ENGINE_ONNX;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;
import static java.awt.Image.SCALE_DEFAULT;

@Slf4j
@Component
@RequiredArgsConstructor
public class YoloUtils {

    private final YoloProperties yolo;

    private ZooModel<Image, DetectedObjects> yoloModel;

    private final Map<Integer, Color> colorMap = Map.of(
            0, new Color(200, 0, 0),
            1, new Color(0, 200, 0),
            2, new Color(0, 0, 200),
            3, new Color(200, 200, 0),
            4, new Color(200, 0, 200),
            5, new Color(0, 200, 200)
    );


    @PostConstruct
    public void init() {
        log.info("开始加载YOLO工具类");

        Device device = Device.Type.CPU.equalsIgnoreCase(yolo.getDeviceType()) ? Device.cpu() : Device.gpu();

        Translator<Image, DetectedObjects> translator = YoloV5Translator
                .builder()
                .optThreshold(yolo.getThreshold())
                .optSynsetArtifactName(yolo.getNameList())
                .build();

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optDevice(device)
                .optModelUrls(getRealUrl(yolo.getYoloUrl()))
                .optModelName(yolo.getModelName())
                .optTranslator(translator)
                .optEngine(ENGINE_ONNX)
                .build();

        try {
            yoloModel = ModelZoo.loadModel(criteria);
        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            log.error("加载模型失败", e);
        }
    }


    /**
     * 对象检测函数
     *
     * @param image 图片，尺寸需满足yolo网络入参大小
     */
    @SneakyThrows
    public DetectedObjects detect(BufferedImage image) {

        if (image.getWidth() != yolo.getWidth() || image.getHeight() != yolo.getHeight()) {
            throw new IllegalArgumentException("图片尺寸错误");
        }

        Image img = ImageFactory.getInstance().fromImage(image);
        long startTime = System.currentTimeMillis();

        //开始检测图片
        DetectedObjects detections;
        try(Predictor<Image, DetectedObjects> predictor = yoloModel.newPredictor()) {
            detections = predictor.predict(img);
        }
        log.info("results: {}", detections);

        log.info(String.format("%.2f", 1000.0 / (System.currentTimeMillis() - startTime)));
        return detections;
    }

    /**
     * 检测并绘制结果
     *
     * @param image 原始图片
     * @return      带有绘制结果的图片
     */
    public BufferedImage drawDetections(BufferedImage image) {
        //将图片大小设置为网络输入要求的大小
        BufferedImage scaledImage = scale(image, yolo.getWidth(), yolo.getHeight());

        DetectedObjects detections = detect(scaledImage);

        //将结果绘制到图片中
        drawDetections(scaledImage, detections);

        return scale(scaledImage, image.getWidth(), image.getHeight());
    }


    /**
     * 图片缩放
     *
     * @param original  原始
     * @param width     宽
     * @param height    高
     * @return          缩放后的图片
     */
    public BufferedImage scale(BufferedImage original, int width, int height) {
        if (width == original.getWidth() && height == original.getHeight()) {
            return original;
        }

        java.awt.Image scaledInstance = original.getScaledInstance(width, height, SCALE_DEFAULT);
        BufferedImage scaledImage = new BufferedImage(width, height, original.getType());
        scaledImage.getGraphics().drawImage(scaledInstance, 0, 0, null);
        return scaledImage;
    }


    /**
     * 绘制检测结果
     *
     * @param image           图片
     * @param detections    检测结果
     */
    private void drawDetections(BufferedImage image, DetectedObjects detections) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        List<DetectedObjects.DetectedObject> list = detections.items();
        for (DetectedObjects.DetectedObject result : list) {
            String className = result.getClassName();
            BoundingBox box = result.getBoundingBox();
            double probability = result.getProbability();
            Color color = colorMap.get(Math.abs(className.hashCode() % 6));
            g.setPaint(color);

            Rectangle rectangle = box.getBounds();
            int x = (int) (rectangle.getX());
            int y = (int) (rectangle.getY());
            int width = (int) (rectangle.getWidth());
            g.drawRect(x, y, width, (int) (rectangle.getHeight()));
            drawText(g, className, probability, x, y, width);
        }
        g.dispose();
    }

    private static void drawText(Graphics2D g, String className, double probability, int x, int y, int width) {
        //设置水印的坐标
        String showText = String.format("种类:%s; 置信度: %.2f%%", className, probability * 100);
        g.fillRect(x, y - 30, width, 30);

        g.setColor(Color.WHITE);
        g.setFont(new Font("微软雅黑", Font.PLAIN, 18));//设置字体
        g.drawString(showText, x, y - 10);
    }

}
