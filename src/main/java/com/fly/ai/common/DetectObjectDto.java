package com.fly.ai.common;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

import java.util.HashMap;
import java.util.Map;

import static java.util.Objects.isNull;

/**
 * @author TinyThing
 * @see <a href="https://www.jianshu.com/u/aba665c4151f">简书TinyThing</a>
 * @since 2022/3/19 19:57
 */
@Data
@Accessors(chain = true)
@NoArgsConstructor
public class DetectObjectDto {
    private String className;
    private Double probability;
    private Double x;
    private Double y;
    private Double width;
    private Double height;
    private Map<String, Object> data;

    public DetectObjectDto(Classifications.Classification item) {
        if (!(item instanceof DetectedObjects.DetectedObject)) {
            throw new IllegalArgumentException("item is not DetectedObject");
        }

        DetectedObjects.DetectedObject i = (DetectedObjects.DetectedObject) item;

        this.className = i.getClassName();
        this.probability = i.getProbability();
        this.x = i.getBoundingBox().getBounds().getX();
        this.y = i.getBoundingBox().getBounds().getY();
        this.width = i.getBoundingBox().getBounds().getWidth();
        this.height = i.getBoundingBox().getBounds().getHeight();
    }

    public Map<String, Object> getData() {
        if (isNull(data)) {
            data = new HashMap<>();
        }
        return data;
    }
}
