create extension if not exists vector; -- enable pgvector extension

create table if not exists public.document
(
    document_id serial
        constraint document_pk
            primary key,
    title       varchar(1024)       not null,
    created_on  timestamp default now(),
    blob_url    varchar(1024)       not null,
    status      integer   default 0 not null,
    chunk_count integer             not null
);

alter table public.document
    owner to postgres;

create table if not exists public.doc_vector
(
    doc_vector_id bigint    default nextval('doc_vector_text_embedding_id_seq'::regclass) not null
        constraint text_embedding_id
            primary key,
    created_on    timestamp default now(),
    text          varchar(2048)                                                           not null,
    embedding     vector(768)                                                             not null,
    document_id   integer                                                                 not null
        constraint fk_doc_vector_document_id_document_document_id
            references public.document,
    chunk_id      varchar(500)
);

alter table public.doc_vector
    owner to postgres;

create unique index if not exists doc_vector_chunk_id_uindex
    on public.doc_vector (chunk_id);

create index if not exists ix_l2_doc_vector_embedding
    on public.doc_vector using hnsw (embedding public.vector_l2_ops);

