package com.fly.ai.yolov8;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import static com.fly.ai.common.Constants.ENGINE_PYTORCH;

/**
 * @author guoxiang
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2023/4/19 17:34
 */
@Data
@ConfigurationProperties(prefix = "ai.fire")
@Configuration
public class FireProperties {

    /**
     * 如果使用gpu，则需要模型导出时增加参数device=0，
     * 目前onnx使用gpu会报错，建议使用其他格式的模型
     */
    private String deviceType = "cpu";

    /**
     * 注意如果打包成jar文件，则需要将模型压缩到zip包里面，然后url使用zip路径，可以参考 YoloProperties
     */
    private String modelUrl = "/model/fire";
    // private String modelName = "best.onnx";
    // private String engine = "OnnxRuntime";


    /**
     * 使用torchscript格式的模型，注意导出模型时要指定device是cpu还是gpu
     */
    private String modelName = "best_cpu.torchscript";
    // private String modelName = "best_gpu.torchscript";
    private String engine = ENGINE_PYTORCH;


    private Float threshold = 0.25f;

    /**
     * 我的模型是800*800的，根据你自己的模型修改
     */
    private Integer width = 800;
    private Integer height = 800;
}
