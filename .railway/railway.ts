import { defineRailway, github, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const tlGdprComplianceRedactionVolume = volume("tl-gdpr-compliance-redaction-volume", { alerts: { usage: { "100": {}, "80": {}, "95": {} } }, allowOnlineResize: true, region: "europe-west4-drams3a", sizeMB: 50000 });
  const tlGDPRComplianceRedaction = service("tl-GDPR-compliance-redaction", {
    source: github("Hrishikesh332/tl-GDPR-compliance-redaction", { branch: "master", checkSuites: false, rootDirectory: "/" }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "Dockerfile", watchPatterns: ["/backend/**", "/frontend/public/generated-thumbnails/**", "/Dockerfile", "/.dockerignore"] },
    healthcheck: "/",
    healthcheckTimeout: 300,
    replicas: { "europe-west4-drams3a": 1 },
    deploy: { restartPolicyType: "ON_FAILURE", restartPolicyMaxRetries: 3, sleepApplication: false },
    networking: { privateNetworkEndpoint: "tl-gdpr-compliance-redaction" },
    volumeMounts: { "/data": tlGdprComplianceRedactionVolume },
    env: { DATA_DIR: "/data", TWELVELABS_API_KEY: preserve(), TWELVELABS_ENTITY_COLLECTION_ID: preserve(), TWELVELABS_INDEX_ID: preserve() },
  });

  return project("inspiring-courage", {
    resources: [tlGDPRComplianceRedaction, tlGdprComplianceRedactionVolume],
  });
});
