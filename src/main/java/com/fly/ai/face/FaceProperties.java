package com.fly.ai.face;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/20 12:22
 */
@Data
@ConfigurationProperties(prefix = "ai.face")
@Configuration
public class FaceProperties {

    /**
     * 置信度阈值
     */
    double confThresh = 0.85f;

    /**
     * nms阈值
     */
    double nmsThresh = 0.45f;

    private String deviceType = "cpu";

    String detectModelUrl = "/model/face/retinaface.zip";

    String featureModelUrl = "/model/face/face_feature.zip";

}
