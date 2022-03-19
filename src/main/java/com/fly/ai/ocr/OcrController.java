package com.fly.ai.ocr;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * ocr识别服务
 *
 * @author 别动我
 * @since 2022/3/16 17:27
 */
@RestController
@RequestMapping("ocr")
@RequiredArgsConstructor
@Slf4j
public class OcrController {

    private final OcrUtils ocrUtils;

    @PostMapping("{type}")
    @ApiOperation("识别图片中的文字并返回结果")
    public List<OcrDto> ocr(@RequestPart MultipartFile image,
                            @PathVariable(required = false) OcrType type) throws IOException {

        DetectedObjects result = ocrUtils.ocr(image.getInputStream(), type);

        return result.items().stream().map(OcrDto::new).collect(Collectors.toList());
    }


    /**
     * 返回文字识别后的图片
     *
     * @param image     原始图片
     * @param type      识别类型，默认QUICK
     * @param response  响应
     * @throws IOException  异常
     */
    @PostMapping("image/{type}")
    @ApiOperation("识别图片中的文字并返回识别后的图片")
    public void ocrImage(@RequestPart MultipartFile image,
                         @PathVariable(required = false) OcrType type,
                         HttpServletResponse response) throws IOException {
        Image img = ImageFactory.getInstance().fromInputStream(image.getInputStream());
        DetectedObjects result = ocrUtils.ocr(img, type);

        result.items().forEach(item -> log.debug(item.getClassName()));

        BufferedImage resultImage = ocrUtils.createResultImage(img, result);

        response.setContentType("image/png");
        ServletOutputStream os = response.getOutputStream();
        ImageIO.write(resultImage, "PNG", os);
        os.flush();
    }

}
