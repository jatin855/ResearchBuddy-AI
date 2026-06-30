package com.researchbuddy.backend.paper;

import com.fasterxml.jackson.databind.JsonNode;
import com.researchbuddy.backend.paper.dto.AiUploadResponse;
import com.researchbuddy.backend.paper.dto.CompareRequest;
import com.researchbuddy.backend.paper.dto.QuestionRequest;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

@Component
public class AiServiceClient {
    private final WebClient client;

    public AiServiceClient(WebClient aiWebClient) {
        this.client = aiWebClient;
    }

    public AiUploadResponse upload(MultipartFile file) {
        try {
            var resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };
            var body = new LinkedMultiValueMap<String, Object>();
            body.add("file", resource);
            return client.post()
                .uri("/papers/upload")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(body))
                .retrieve()
                .bodyToMono(AiUploadResponse.class)
                .block();
        } catch (Exception ex) {
            throw new IllegalStateException("AI upload failed", ex);
        }
    }

    public JsonNode summary(String paperId) {
        return client.get()
            .uri("/papers/{paperId}/summary", paperId)
            .retrieve()
            .bodyToMono(JsonNode.class)
            .block();
    }

    public JsonNode ask(String paperId, QuestionRequest request) {
        return client.post()
            .uri("/papers/{paperId}/qa", paperId)
            .bodyValue(request)
            .retrieve()
            .bodyToMono(JsonNode.class)
            .block();
    }

    public JsonNode compare(CompareRequest request) {
        return client.post()
            .uri("/papers/compare")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(JsonNode.class)
            .block();
    }
}
