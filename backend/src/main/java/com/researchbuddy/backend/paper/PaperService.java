package com.researchbuddy.backend.paper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.researchbuddy.backend.paper.dto.CompareRequest;
import com.researchbuddy.backend.paper.dto.QuestionRequest;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class PaperService {
    private final PaperRepository repository;
    private final AiServiceClient ai;
    private final ObjectMapper mapper;

    public PaperService(PaperRepository repository, AiServiceClient ai, ObjectMapper mapper) {
        this.repository = repository;
        this.ai = ai;
        this.mapper = mapper;
    }

    public Paper upload(MultipartFile file) {
        var response = ai.upload(file);
        var paper = new Paper();
        paper.setId(response.paperId());
        paper.setTitle(response.title());
        paper.setFileName(response.fileName());
        paper.setStatus("READY");
        return repository.save(paper);
    }

    public List<Paper> list() {
        return repository.findAll();
    }

    public Paper get(String id) {
        return repository.findById(id).orElseThrow(() -> new PaperNotFoundException(id));
    }

    public JsonNode summary(String id) {
        get(id);
        var result = ai.summary(id);
        repository.findById(id).ifPresent(paper -> {
            try {
                paper.setSummaryJson(mapper.writeValueAsString(result));
                repository.save(paper);
            } catch (Exception ignored) {
            }
        });
        return result;
    }

    public JsonNode ask(String id, QuestionRequest request) {
        get(id);
        return ai.ask(id, request);
    }

    public JsonNode compare(CompareRequest request) {
        request.paperIds().forEach(this::get);
        return ai.compare(request);
    }
}
