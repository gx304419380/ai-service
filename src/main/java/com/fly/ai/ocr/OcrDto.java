package com.fly.ai.ocr;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

/**
 * @author 别动我
 * @since 2022/3/16 17:39
 */
@Data
@NoArgsConstructor
@Accessors(chain = true)
public class OcrDto {
    private String content;
    private Double x;
    private Double y;
    private Double width;
    private Double height;


    public OcrDto(Classifications.Classification item) {
        if (!(item instanceof DetectedObjects.DetectedObject)) {
            throw new IllegalArgumentException("item is not DetectedObject");
        }

        DetectedObjects.DetectedObject i = (DetectedObjects.DetectedObject) item;

        this.content = i.getClassName();
        this.x = i.getBoundingBox().getBounds().getX();
        this.y = i.getBoundingBox().getBounds().getY();
        this.width = i.getBoundingBox().getBounds().getWidth();
        this.height = i.getBoundingBox().getBounds().getHeight();
    }
}
