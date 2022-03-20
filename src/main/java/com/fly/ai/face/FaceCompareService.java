package com.fly.ai.face;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.fly.ai.common.DetectObjectDto;
import com.fly.ai.common.ImageUtils;
import com.fly.ai.common.PredictException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.fly.ai.common.Constants.ENGINE_PYTORCH;
import static com.fly.ai.common.ModelUrlUtils.getRealUrl;

/**
 * 人脸比对服务
 *
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/20 13:53
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class FaceCompareService {

    private final FaceProperties prop;
    private final FaceDetectService detectService;

    private ZooModel<Image, float[]> model;

    @PostConstruct
    public void init() throws ModelNotFoundException, MalformedModelException, IOException {
        log.info("开始加载人脸特征提取模型");

        Device device = Device.Type.CPU.equalsIgnoreCase(prop.getDeviceType()) ? Device.cpu() : Device.gpu();

        Criteria<Image, float[]> criteria = Criteria.builder()
                .setTypes(Image.class, float[].class)
                .optDevice(device)
                .optModelUrls(getRealUrl(prop.getFeatureModelUrl()))
                .optTranslator(new FaceFeatureTranslator())
                .optProgress(new ProgressBar())
                .optEngine(ENGINE_PYTORCH)
                .build();

        model = criteria.loadModel();

        log.info("完成加载人脸特征提取模型");
    }

    /**
     * 人脸特征值提取
     *
     * @param image     image
     * @return          特征值向量
     */
    public float[] feature(Image image) {
        try (Predictor<Image, float[]> predictor = model.newPredictor()) {
            return feature(image, predictor);
        }
    }

    /**
     * 人脸特征值提取
     *
     * @param image     image
     * @param predictor 循环中使用，避免频繁new
     * @return          特征值向量
     */
    public float[] feature(Image image, Predictor<Image, float[]> predictor) {
        long start = System.currentTimeMillis();

        //图片宽高小于112
//        image = formatImage(image);

        float[] feature;
        try {
            feature = predictor.predict(image);
        } catch (TranslateException e) {
            throw new PredictException("人脸提取特征值错误", e);
        }

        log.debug("人脸特征提取耗时{}ms", System.currentTimeMillis() - start);
        return feature;
    }


    /**
     * 图片大小有要求，小于112
     *
     * @param image 图片
     * @return      个格式化后的图片
     */
    private Image formatImage(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();

        log.debug("w = {}, h = {}", width, height);
        if (width >= height) {
            if (width <= 112) {
                return image;
            }

            return ImageUtils.scale(image, 112, (int) (height * 112.0 / width));
        }
        if (height <= 112) {
            return image;
        }
        return ImageUtils.scale(image, (int) (width * 112.0 / height), 112);
    }

    /**
     * 获取人脸特征值list
     *
     * @param image 图片
     * @return      图中各个人脸特征值
     */
    public List<DetectObjectDto> getFeatureList(Image image) {

        DetectedObjects detection = detectService.detect(image);
        List<DetectedObjects.DetectedObject> items = detection.items();
        List<DetectObjectDto> list = new ArrayList<>(items.size());


        try(Predictor<Image, float[]> predictor = model.newPredictor()) {
            //截取各个人脸图片，计算特征值
            for (DetectedObjects.DetectedObject face : items) {
                Image subImage = ImageUtils.getSubImage(image, face.getBoundingBox());
                float[] predict = feature(subImage, predictor);
                DetectObjectDto dto = new DetectObjectDto(face);
                dto.getData().put("feature", predict);
                list.add(dto);
            }
        }

        return list;
    }


    public static float similar(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }
}
