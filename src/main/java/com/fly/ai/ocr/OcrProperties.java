package com.fly.ai.ocr;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * OCR配置
 *
 * @author 别动我
 * @since 2022/3/16 22:07
 */
@Data
@ConfigurationProperties(prefix = "ai.ocr")
@Configuration
public class OcrProperties {

    /**
     * 旋转判断阈值，default=0.8
     */
    private double rotateThreshold = 0.8;

    private String detectUrl = "/model/ocr/detect.zip";

    private String detectQuickUrl = "/model/ocr/detect_mobile.zip";

    private String recognizeUrl = "/model/ocr/recognize.zip";

    private String recognizeQuickUrl = "/model/ocr/recognize_mobile.zip";

    private String rotateUrl = "/model/ocr/rotate.zip";

    private String deviceType = "cpu";


}
