package com.researchbuddy.backend.paper;

public class PaperNotFoundException extends RuntimeException {
    public PaperNotFoundException(String id) {
        super("Paper not found: " + id);
    }
}
