#pragma once
#include <llama.h>
#include <QMutex>
#include <QDebug>


namespace QtLlama {
   inline void ensureBackendInit() {
        static QBasicMutex mutex;
        static bool initialized = false;
        QMutexLocker lock(&mutex);
        if (!initialized) {
            llama_backend_init();
            initialized = true;
        }

        llama_log_set([](enum ggml_log_level level, const char* text, void*) {
            switch (level) {
                case GGML_LOG_LEVEL_ERROR: qCritical() << "[llama.cpp]" << text; break;
                case GGML_LOG_LEVEL_WARN:  qWarning()  << "[llama.cpp]" << text; break;
                default: break; // suppress info/debug by default
            }
        }, nullptr);
    } 
}
