package com.fly.ai;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

class AiServiceApplicationTests {

    @Test
    void contextLoads() {
        String env = System.getenv("DJL_CACHE_DIR");
        System.out.println(env);
    }

}
