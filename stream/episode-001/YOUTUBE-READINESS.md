# YouTube readiness gate

Episode 001 is not publishable until every item is checked from direct evidence.

- [x] All three simulator challenges passed their machine-readable gates.
- [x] Result measurements match the video overlays and metadata.
- [x] A privacy-safe 1920×1080 visual master was rendered and inspected.
- [x] A 1280×720 thumbnail was rendered and inspected.
- [x] Title, description, repository link, attribution, and restrained search phrases are drafted.
- [x] A genuine-human narration script and recording notes are ready.
- [ ] Genuine human narration has been received; no generated or cloned voice is present.
- [ ] Narration has been cleaned, normalized, and mixed into the visual master.
- [ ] Captions have been aligned to the real human delivery.
- [ ] The complete final video has been watched and listened to end-to-end.
- [ ] Final privacy review found no personal identity, contact information, credentials, notifications, or unrelated projects.
- [ ] User has given action-time approval for the exact YouTube upload.

Only after the first ten production checks pass may `stream/production-status.json`
be changed to `READY_FOR_YOUTUBE`. Publication remains a separate approved action.
