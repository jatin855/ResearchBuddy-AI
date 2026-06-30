package com.researchbuddy.backend.paper;

import com.fasterxml.jackson.databind.JsonNode;
import com.researchbuddy.backend.paper.dto.CompareRequest;
import com.researchbuddy.backend.paper.dto.QuestionRequest;
import java.util.List;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class PaperController {
    private final PaperService service;

    public PaperController(PaperService service) {
        this.service = service;
    }

    @PostMapping(value = "/papers", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Paper upload(@RequestPart("file") MultipartFile file) {
        return service.upload(file);
    }

    @GetMapping("/papers")
    public List<Paper> papers() {
        return service.list();
    }

    @GetMapping("/papers/{id}")
    public Paper paper(@PathVariable String id) {
        return service.get(id);
    }

    @GetMapping("/papers/{id}/summary")
    public JsonNode summary(@PathVariable String id) {
        return service.summary(id);
    }

    @PostMapping("/papers/{id}/qa")
    public JsonNode ask(@PathVariable String id, @RequestBody QuestionRequest request) {
        return service.ask(id, request);
    }

    @PostMapping("/papers/compare")
    public JsonNode compare(@RequestBody CompareRequest request) {
        return service.compare(request);
    }
}
