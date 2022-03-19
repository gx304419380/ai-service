package com.fly.ai.yolo;

import ai.djl.modality.cv.output.DetectedObjects;
import com.fly.ai.ocr.OcrDto;
import com.fly.ai.ocr.OcrType;
import com.fly.ai.ocr.OcrUtils;
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
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 13:38
 */
@RestController
@RequestMapping("yolo")
@RequiredArgsConstructor
@Slf4j
public class YoloController {

    private final YoloUtils yoloUtils;

    @PostMapping()
    @ApiOperation("图片对相关检测")
    public void ocr(@RequestPart MultipartFile file, HttpServletResponse response) throws IOException {

        BufferedImage image = ImageIO.read(file.getInputStream());
        BufferedImage result = yoloUtils.drawDetections(image);

        response.setContentType("image/png");
        ServletOutputStream os = response.getOutputStream();
        ImageIO.write(result, "PNG", os);
        os.flush();
    }


}
