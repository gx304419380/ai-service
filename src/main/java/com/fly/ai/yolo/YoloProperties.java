package com.fly.ai.yolo;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * @author tiny tiny
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 12:02
 */
@Data
@ConfigurationProperties(prefix = "ai.yolo")
@Configuration
public class YoloProperties {

    private String deviceType = "cpu";

    private String yoloUrl = "/model/yolo";

    /**
     * 默认使用中等模型
     */
    private String modelName = "yolov5m.onnx";

    private String nameList = "coco.names";

    private Float threshold = 0.5f;

    private Integer width = 640;
    private Integer height = 640;

}
