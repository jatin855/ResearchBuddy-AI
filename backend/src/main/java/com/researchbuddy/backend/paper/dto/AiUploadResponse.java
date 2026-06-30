package com.researchbuddy.backend.paper.dto;

public record AiUploadResponse(
    String paperId,
    String title,
    String fileName,
    int sectionCount,
    int chunkCount
) {
}
