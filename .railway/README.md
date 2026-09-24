# Railway infrastructure

The `railway.ts` file records the existing production backend service and its
persistent `/data` volume. Secret values stay in Railway via `preserve()`.

With Node 22 or newer, run from the repository root:

```sh
npm ci --prefix .railway
npx @railway/cli link --project e5a85239-5bdb-48e7-a717-9129c0106340 --environment production --service tl-GDPR-compliance-redaction
npx @railway/cli config plan
npx @railway/cli config apply
```

Review the plan before applying. Preserve the existing volume: do not delete,
shrink, or detach it to deploy code. Ordinary code pushes to `master` trigger
the GitHub-connected backend build. Changes under `.railway/` need the explicit
plan/apply workflow; Railway does not read this directory during code builds.

The image is built with the repository-root Dockerfile. Keep one replica and
one Gunicorn worker because active job state lives in memory. The app listens
on Railway's `PORT`, uses `/` for health checks, and does not sleep when idle.

The generated Railway domain is managed separately. The current application
has no authentication; review access requirements before enabling public use.
